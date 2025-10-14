{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1bQs2mQaVxU2x6iCTSrnsxXjIzUISqQGC",
      "authorship_tag": "ABX9TyOQgh5wrdBrkyqEVzYPFdUj",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/48harry/Smart-Shipping-Logistics-Abnormal-Operation-Diagnosis-Based-on-Abnormal-Signal-Detection/blob/main/training_pkl_1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "qekR-Q_dgLLk"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import lightgbm as lgb\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import roc_auc_score\n",
        "from tqdm import tqdm\n",
        "import joblib\n",
        "tqdm.pandas()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "BASE_DIR = \"/content/drive/MyDrive/dacon_driver/\"\n",
        "\n",
        "train_meta = pd.read_csv(os.path.join(BASE_DIR, \"data/train.csv\"))\n",
        "train_A = pd.read_csv(os.path.join(BASE_DIR, \"data/train/A.csv\"))\n",
        "train_B = pd.read_csv(os.path.join(BASE_DIR, \"data/train/B.csv\"))\n",
        "\n",
        "print(\"train_meta:\", train_meta.shape)\n",
        "print(\"train_A:\", train_A.shape)\n",
        "print(\"train_B:\", train_B.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hS66xD_-gQYO",
        "outputId": "62fba1da-14ed-4f99-d81a-c1fdfd0154b6"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "train_meta: (944767, 3)\n",
            "train_A: (647241, 37)\n",
            "train_B: (297526, 31)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def convert_age(val):\n",
        "    if pd.isna(val): return np.nan\n",
        "    try:\n",
        "        base = int(str(val)[:-1])\n",
        "        return base if str(val)[-1] == \"a\" else base + 5\n",
        "    except:\n",
        "        return np.nan\n",
        "\n",
        "def split_testdate(val):\n",
        "    try:\n",
        "        v = int(val)\n",
        "        return v // 100, v % 100\n",
        "    except:\n",
        "        return np.nan, np.nan\n",
        "\n",
        "def seq_mean(series):\n",
        "    return series.fillna(\"\").progress_apply(\n",
        "        lambda x: np.fromstring(x, sep=\",\").mean() if x else np.nan\n",
        "    )\n",
        "\n",
        "def seq_std(series):\n",
        "    return series.fillna(\"\").progress_apply(\n",
        "        lambda x: np.fromstring(x, sep=\",\").std() if x else np.nan\n",
        "    )\n",
        "\n",
        "def seq_rate(series, target=\"1\"):\n",
        "    return series.fillna(\"\").progress_apply(\n",
        "        lambda x: str(x).split(\",\").count(target) / len(x.split(\",\")) if x else np.nan\n",
        "    )\n",
        "\n",
        "def masked_mean_from_csv_series(cond_series, val_series, mask_val):\n",
        "    cond_df = cond_series.fillna(\"\").str.split(\",\", expand=True).replace(\"\", np.nan)\n",
        "    val_df  = val_series.fillna(\"\").str.split(\",\", expand=True).replace(\"\", np.nan)\n",
        "\n",
        "    cond_arr = cond_df.to_numpy(dtype=float)\n",
        "    val_arr  = val_df.to_numpy(dtype=float)\n",
        "\n",
        "    mask = (cond_arr == mask_val)\n",
        "    with np.errstate(invalid=\"ignore\"):\n",
        "        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)\n",
        "        counts = np.sum(mask, axis=1)\n",
        "        out = sums / np.where(counts==0, np.nan, counts)\n",
        "    return pd.Series(out, index=cond_series.index)\n",
        "\n",
        "def masked_mean_in_set_series(cond_series, val_series, mask_set):\n",
        "    cond_df = cond_series.fillna(\"\").str.split(\",\", expand=True).replace(\"\", np.nan)\n",
        "    val_df  = val_series.fillna(\"\").str.split(\",\", expand=True).replace(\"\", np.nan)\n",
        "\n",
        "    cond_arr = cond_df.to_numpy(dtype=float)\n",
        "    val_arr  = val_df.to_numpy(dtype=float)\n",
        "\n",
        "    mask = np.isin(cond_arr, list(mask_set))\n",
        "    with np.errstate(invalid=\"ignore\"):\n",
        "        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)\n",
        "        counts = np.sum(mask, axis=1)\n",
        "        out = sums / np.where(counts == 0, np.nan, counts)\n",
        "    return pd.Series(out, index=cond_series.index)"
      ],
      "metadata": {
        "id": "Voe4Wtx3gTcQ"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def preprocess_A(train_A):\n",
        "    df = train_A.copy()\n",
        "\n",
        "    # ---- Age, TestDate 파생 ----\n",
        "    print(\"Step 1: Age, TestDate 파생...\")\n",
        "    df[\"Age_num\"] = df[\"Age\"].map(convert_age)\n",
        "    ym = df[\"TestDate\"].map(split_testdate)\n",
        "    df[\"Year\"] = [y for y, m in ym]\n",
        "    df[\"Month\"] = [m for y, m in ym]\n",
        "\n",
        "    feats = pd.DataFrame(index=df.index)\n",
        "\n",
        "    # ---- A1 ----\n",
        "    print(\"Step 2: A1 feature 생성...\")\n",
        "    feats[\"A1_resp_rate\"] = seq_rate(df[\"A1-3\"], \"1\")\n",
        "    feats[\"A1_rt_mean\"]   = seq_mean(df[\"A1-4\"])\n",
        "    feats[\"A1_rt_std\"]    = seq_std(df[\"A1-4\"])\n",
        "    feats[\"A1_rt_left\"]   = masked_mean_from_csv_series(df[\"A1-1\"], df[\"A1-4\"], 1)\n",
        "    feats[\"A1_rt_right\"]  = masked_mean_from_csv_series(df[\"A1-1\"], df[\"A1-4\"], 2)\n",
        "    feats[\"A1_rt_side_diff\"] = feats[\"A1_rt_left\"] - feats[\"A1_rt_right\"]\n",
        "    feats[\"A1_rt_slow\"]   = masked_mean_from_csv_series(df[\"A1-2\"], df[\"A1-4\"], 1)\n",
        "    feats[\"A1_rt_fast\"]   = masked_mean_from_csv_series(df[\"A1-2\"], df[\"A1-4\"], 3)\n",
        "    feats[\"A1_rt_speed_diff\"] = feats[\"A1_rt_slow\"] - feats[\"A1_rt_fast\"]\n",
        "\n",
        "    # ---- A2 ----\n",
        "    print(\"Step 3: A2 feature 생성...\")\n",
        "    feats[\"A2_resp_rate\"] = seq_rate(df[\"A2-3\"], \"1\")\n",
        "    feats[\"A2_rt_mean\"]   = seq_mean(df[\"A2-4\"])\n",
        "    feats[\"A2_rt_std\"]    = seq_std(df[\"A2-4\"])\n",
        "    feats[\"A2_rt_cond1_diff\"] = masked_mean_from_csv_series(df[\"A2-1\"], df[\"A2-4\"], 1) - \\\n",
        "                                masked_mean_from_csv_series(df[\"A2-1\"], df[\"A2-4\"], 3)\n",
        "    feats[\"A2_rt_cond2_diff\"] = masked_mean_from_csv_series(df[\"A2-2\"], df[\"A2-4\"], 1) - \\\n",
        "                                masked_mean_from_csv_series(df[\"A2-2\"], df[\"A2-4\"], 3)\n",
        "\n",
        "    # ---- A3 ----\n",
        "    print(\"Step 4: A3 feature 생성...\")\n",
        "    s = df[\"A3-5\"].fillna(\"\")\n",
        "    total   = s.apply(lambda x: len(x.split(\",\")) if x else 0)\n",
        "    valid   = s.apply(lambda x: sum(v in {\"1\",\"2\"} for v in x.split(\",\")) if x else 0)\n",
        "    invalid = s.apply(lambda x: sum(v in {\"3\",\"4\"} for v in x.split(\",\")) if x else 0)\n",
        "    correct = s.apply(lambda x: sum(v in {\"1\",\"3\"} for v in x.split(\",\")) if x else 0)\n",
        "    feats[\"A3_valid_ratio\"]   = (valid / total).replace([np.inf,-np.inf], np.nan)\n",
        "    feats[\"A3_invalid_ratio\"] = (invalid / total).replace([np.inf,-np.inf], np.nan)\n",
        "    feats[\"A3_correct_ratio\"] = (correct / total).replace([np.inf,-np.inf], np.nan)\n",
        "\n",
        "    feats[\"A3_resp2_rate\"] = seq_rate(df[\"A3-6\"], \"1\")\n",
        "    feats[\"A3_rt_mean\"]    = seq_mean(df[\"A3-7\"])\n",
        "    feats[\"A3_rt_std\"]     = seq_std(df[\"A3-7\"])\n",
        "    feats[\"A3_rt_size_diff\"] = masked_mean_from_csv_series(df[\"A3-1\"], df[\"A3-7\"], 1) - \\\n",
        "                               masked_mean_from_csv_series(df[\"A3-1\"], df[\"A3-7\"], 2)\n",
        "    feats[\"A3_rt_side_diff\"] = masked_mean_from_csv_series(df[\"A3-3\"], df[\"A3-7\"], 1) - \\\n",
        "                               masked_mean_from_csv_series(df[\"A3-3\"], df[\"A3-7\"], 2)\n",
        "\n",
        "    # ---- A4 ----\n",
        "    print(\"Step 5: A4 feature 생성...\")\n",
        "    feats[\"A4_acc_rate\"]   = seq_rate(df[\"A4-3\"], \"1\")\n",
        "    feats[\"A4_resp2_rate\"] = seq_rate(df[\"A4-4\"], \"1\")\n",
        "    feats[\"A4_rt_mean\"]    = seq_mean(df[\"A4-5\"])\n",
        "    feats[\"A4_rt_std\"]     = seq_std(df[\"A4-5\"])\n",
        "    feats[\"A4_stroop_diff\"] = masked_mean_from_csv_series(df[\"A4-1\"], df[\"A4-5\"], 2) - \\\n",
        "                              masked_mean_from_csv_series(df[\"A4-1\"], df[\"A4-5\"], 1)\n",
        "    feats[\"A4_rt_color_diff\"] = masked_mean_from_csv_series(df[\"A4-2\"], df[\"A4-5\"], 1) - \\\n",
        "                                masked_mean_from_csv_series(df[\"A4-2\"], df[\"A4-5\"], 2)\n",
        "\n",
        "    # ---- A5 ----\n",
        "    print(\"Step 6: A5 feature 생성...\")\n",
        "    feats[\"A5_acc_rate\"]   = seq_rate(df[\"A5-2\"], \"1\")\n",
        "    feats[\"A5_resp2_rate\"] = seq_rate(df[\"A5-3\"], \"1\")\n",
        "    feats[\"A5_acc_nonchange\"] = masked_mean_from_csv_series(df[\"A5-1\"], df[\"A5-2\"], 1)\n",
        "    feats[\"A5_acc_change\"]    = masked_mean_in_set_series(df[\"A5-1\"], df[\"A5-2\"], {2,3,4})\n",
        "\n",
        "    # ---- Drop ----\n",
        "    print(\"Step 7: 시퀀스 컬럼 drop & concat...\")\n",
        "    seq_cols = [\n",
        "        \"A1-1\",\"A1-2\",\"A1-3\",\"A1-4\",\n",
        "        \"A2-1\",\"A2-2\",\"A2-3\",\"A2-4\",\n",
        "        \"A3-1\",\"A3-2\",\"A3-3\",\"A3-4\",\"A3-5\",\"A3-6\",\"A3-7\",\n",
        "        \"A4-1\",\"A4-2\",\"A4-3\",\"A4-4\",\"A4-5\",\n",
        "        \"A5-1\",\"A5-2\",\"A5-3\"\n",
        "    ]\n",
        "    print(\"A 검사 데이터 전처리 완료\")\n",
        "    return pd.concat([df.drop(columns=seq_cols, errors=\"ignore\"), feats], axis=1)"
      ],
      "metadata": {
        "id": "_4EK_h7nhvv9"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def preprocess_B(train_B):\n",
        "    df = train_B.copy()\n",
        "\n",
        "    # ---- Age, TestDate ----\n",
        "    print(\"Step 1: Age, TestDate 파생...\")\n",
        "    df[\"Age_num\"] = df[\"Age\"].map(convert_age)\n",
        "    ym = df[\"TestDate\"].map(split_testdate)\n",
        "    df[\"Year\"] = [y for y, m in ym]\n",
        "    df[\"Month\"] = [m for y, m in ym]\n",
        "\n",
        "    feats = pd.DataFrame(index=df.index)\n",
        "\n",
        "    # ---- B1 ----\n",
        "    print(\"Step 2: B1 feature 생성...\")\n",
        "    feats[\"B1_acc_task1\"] = seq_rate(df[\"B1-1\"], \"1\")\n",
        "    feats[\"B1_rt_mean\"]   = seq_mean(df[\"B1-2\"])\n",
        "    feats[\"B1_rt_std\"]    = seq_std(df[\"B1-2\"])\n",
        "    feats[\"B1_acc_task2\"] = seq_rate(df[\"B1-3\"], \"1\")\n",
        "\n",
        "    # ---- B2 ----\n",
        "    print(\"Step 3: B2 feature 생성...\")\n",
        "    feats[\"B2_acc_task1\"] = seq_rate(df[\"B2-1\"], \"1\")\n",
        "    feats[\"B2_rt_mean\"]   = seq_mean(df[\"B2-2\"])\n",
        "    feats[\"B2_rt_std\"]    = seq_std(df[\"B2-2\"])\n",
        "    feats[\"B2_acc_task2\"] = seq_rate(df[\"B2-3\"], \"1\")\n",
        "\n",
        "    # ---- B3 ----\n",
        "    print(\"Step 4: B3 feature 생성...\")\n",
        "    feats[\"B3_acc_rate\"] = seq_rate(df[\"B3-1\"], \"1\")\n",
        "    feats[\"B3_rt_mean\"]  = seq_mean(df[\"B3-2\"])\n",
        "    feats[\"B3_rt_std\"]   = seq_std(df[\"B3-2\"])\n",
        "\n",
        "    # ---- B4 ----\n",
        "    print(\"Step 5: B4 feature 생성...\")\n",
        "    feats[\"B4_acc_rate\"] = seq_rate(df[\"B4-1\"], \"1\")\n",
        "    feats[\"B4_rt_mean\"]  = seq_mean(df[\"B4-2\"])\n",
        "    feats[\"B4_rt_std\"]   = seq_std(df[\"B4-2\"])\n",
        "\n",
        "    # ---- B5 ----\n",
        "    print(\"Step 6: B5 feature 생성...\")\n",
        "    feats[\"B5_acc_rate\"] = seq_rate(df[\"B5-1\"], \"1\")\n",
        "    feats[\"B5_rt_mean\"]  = seq_mean(df[\"B5-2\"])\n",
        "    feats[\"B5_rt_std\"]   = seq_std(df[\"B5-2\"])\n",
        "\n",
        "    # ---- B6~B8 ----\n",
        "    print(\"Step 7: B6~B8 feature 생성...\")\n",
        "    feats[\"B6_acc_rate\"] = seq_rate(df[\"B6\"], \"1\")\n",
        "    feats[\"B7_acc_rate\"] = seq_rate(df[\"B7\"], \"1\")\n",
        "    feats[\"B8_acc_rate\"] = seq_rate(df[\"B8\"], \"1\")\n",
        "\n",
        "    # ---- Drop ----\n",
        "    print(\"Step 8: 시퀀스 컬럼 drop & concat...\")\n",
        "    seq_cols = [\n",
        "        \"B1-1\",\"B1-2\",\"B1-3\",\n",
        "        \"B2-1\",\"B2-2\",\"B2-3\",\n",
        "        \"B3-1\",\"B3-2\",\n",
        "        \"B4-1\",\"B4-2\",\n",
        "        \"B5-1\",\"B5-2\",\n",
        "        \"B6\",\"B7\",\"B8\"\n",
        "    ]\n",
        "\n",
        "    print(\"B 검사 데이터 전처리 완료\")\n",
        "    return pd.concat([df.drop(columns=seq_cols, errors=\"ignore\"), feats], axis=1)"
      ],
      "metadata": {
        "id": "Thczs0H_hylM"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_A_features = preprocess_A(train_A)\n",
        "train_B_features = preprocess_B(train_B)\n",
        "\n",
        "print(\"A:\", train_A_features.shape, \"B:\", train_B_features.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FJ-5Z_dlh2TH",
        "outputId": "5d838345-f8b3-45b7-b222-2e2dd2338aa7"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 1: Age, TestDate 파생...\n",
            "Step 2: A1 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 647241/647241 [00:03<00:00, 209825.94it/s]\n",
            "100%|██████████| 647241/647241 [00:09<00:00, 65202.75it/s]\n",
            "100%|██████████| 647241/647241 [00:24<00:00, 26394.10it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 3: A2 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 647241/647241 [00:02<00:00, 235339.25it/s]\n",
            "100%|██████████| 647241/647241 [00:07<00:00, 85649.04it/s]\n",
            "100%|██████████| 647241/647241 [00:21<00:00, 29693.43it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 4: A3 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 647241/647241 [00:03<00:00, 186802.11it/s]\n",
            "100%|██████████| 647241/647241 [00:10<00:00, 62250.82it/s]\n",
            "100%|██████████| 647241/647241 [00:23<00:00, 27530.28it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 5: A4 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 647241/647241 [00:03<00:00, 172338.25it/s]\n",
            "100%|██████████| 647241/647241 [00:03<00:00, 179855.01it/s]\n",
            "100%|██████████| 647241/647241 [00:15<00:00, 41220.70it/s]\n",
            "100%|██████████| 647241/647241 [00:29<00:00, 22019.43it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 6: A5 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 647241/647241 [00:01<00:00, 405045.14it/s]\n",
            "100%|██████████| 647241/647241 [00:01<00:00, 350674.13it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 7: 시퀀스 컬럼 drop & concat...\n",
            "A 검사 데이터 전처리 완료\n",
            "Step 1: Age, TestDate 파생...\n",
            "Step 2: B1 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 297526/297526 [00:00<00:00, 619242.33it/s]\n",
            "100%|██████████| 297526/297526 [00:03<00:00, 95072.58it/s]\n",
            "100%|██████████| 297526/297526 [00:10<00:00, 28218.99it/s]\n",
            "100%|██████████| 297526/297526 [00:00<00:00, 567357.39it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 3: B2 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 297526/297526 [00:00<00:00, 600307.53it/s]\n",
            "100%|██████████| 297526/297526 [00:03<00:00, 93763.21it/s]\n",
            "100%|██████████| 297526/297526 [00:10<00:00, 28731.31it/s]\n",
            "100%|██████████| 297526/297526 [00:00<00:00, 581922.92it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 4: B3 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 297526/297526 [00:00<00:00, 679877.14it/s]\n",
            "100%|██████████| 297526/297526 [00:03<00:00, 94502.74it/s]\n",
            "100%|██████████| 297526/297526 [00:10<00:00, 29201.39it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 5: B4 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 297526/297526 [00:01<00:00, 239910.61it/s]\n",
            "100%|██████████| 297526/297526 [00:07<00:00, 40590.41it/s]\n",
            "100%|██████████| 297526/297526 [00:13<00:00, 21374.35it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 6: B5 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 297526/297526 [00:00<00:00, 334722.07it/s]\n",
            "100%|██████████| 297526/297526 [00:03<00:00, 83941.52it/s]\n",
            "100%|██████████| 297526/297526 [00:10<00:00, 27186.42it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 7: B6~B8 feature 생성...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 297526/297526 [00:00<00:00, 497885.24it/s]\n",
            "100%|██████████| 297526/297526 [00:00<00:00, 604391.58it/s]\n",
            "100%|██████████| 297526/297526 [00:00<00:00, 660280.60it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 8: 시퀀스 컬럼 drop & concat...\n",
            "B 검사 데이터 전처리 완료\n",
            "A: (647241, 49) B: (297526, 39)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# -------- 공통 유틸 --------\n",
        "def _has(df, cols):  # 필요한 컬럼이 모두 있는지\n",
        "    return all(c in df.columns for c in cols)\n",
        "\n",
        "def _safe_div(a, b, eps=1e-6):\n",
        "    return a / (b + eps)\n",
        "\n",
        "# -------- A 파생 --------\n",
        "def add_features_A(df: pd.DataFrame) -> pd.DataFrame:\n",
        "    feats = df.copy()\n",
        "    eps = 1e-6\n",
        "\n",
        "    # 0) Year-Month 단일축\n",
        "    if _has(feats, [\"Year\",\"Month\"]):\n",
        "        feats[\"YearMonthIndex\"] = feats[\"Year\"] * 12 + feats[\"Month\"]\n",
        "\n",
        "    # 1) 속도-정확도 트레이드오프\n",
        "    if _has(feats, [\"A1_rt_mean\",\"A1_resp_rate\"]):\n",
        "        feats[\"A1_speed_acc_tradeoff\"] = _safe_div(feats[\"A1_rt_mean\"], feats[\"A1_resp_rate\"], eps)\n",
        "    if _has(feats, [\"A2_rt_mean\",\"A2_resp_rate\"]):\n",
        "        feats[\"A2_speed_acc_tradeoff\"] = _safe_div(feats[\"A2_rt_mean\"], feats[\"A2_resp_rate\"], eps)\n",
        "    if _has(feats, [\"A4_rt_mean\",\"A4_acc_rate\"]):\n",
        "        feats[\"A4_speed_acc_tradeoff\"] = _safe_div(feats[\"A4_rt_mean\"], feats[\"A4_acc_rate\"], eps)\n",
        "\n",
        "    # 2) RT 변동계수(CV)\n",
        "    for k in [\"A1\",\"A2\",\"A3\",\"A4\"]:\n",
        "        m, s = f\"{k}_rt_mean\", f\"{k}_rt_std\"\n",
        "        if _has(feats, [m, s]):\n",
        "            feats[f\"{k}_rt_cv\"] = _safe_div(feats[s], feats[m], eps)\n",
        "\n",
        "    # 3) 조건 차이 절댓값(편향 크기)\n",
        "    for name, base in [\n",
        "        (\"A1_rt_side_gap_abs\",  \"A1_rt_side_diff\"),\n",
        "        (\"A1_rt_speed_gap_abs\", \"A1_rt_speed_diff\"),\n",
        "        (\"A2_rt_cond1_gap_abs\", \"A2_rt_cond1_diff\"),\n",
        "        (\"A2_rt_cond2_gap_abs\", \"A2_rt_cond2_diff\"),\n",
        "        (\"A4_stroop_gap_abs\",   \"A4_stroop_diff\"),\n",
        "        (\"A4_color_gap_abs\",    \"A4_rt_color_diff\"),\n",
        "    ]:\n",
        "        if base in feats.columns:\n",
        "            feats[name] = feats[base].abs()\n",
        "\n",
        "    # 4) 정확도 패턴 심화\n",
        "    if _has(feats, [\"A3_valid_ratio\",\"A3_invalid_ratio\"]):\n",
        "        feats[\"A3_valid_invalid_gap\"] = feats[\"A3_valid_ratio\"] - feats[\"A3_invalid_ratio\"]\n",
        "    if _has(feats, [\"A3_correct_ratio\",\"A3_invalid_ratio\"]):\n",
        "        feats[\"A3_correct_invalid_gap\"] = feats[\"A3_correct_ratio\"] - feats[\"A3_invalid_ratio\"]\n",
        "    if _has(feats, [\"A5_acc_change\",\"A5_acc_nonchange\"]):\n",
        "        feats[\"A5_change_nonchange_gap\"] = feats[\"A5_acc_change\"] - feats[\"A5_acc_nonchange\"]\n",
        "\n",
        "    # 5) 간단 메타 리스크 스코어(휴리스틱)\n",
        "    parts = []\n",
        "    if \"A4_stroop_gap_abs\" in feats: parts.append(0.30 * feats[\"A4_stroop_gap_abs\"].fillna(0))\n",
        "    if \"A4_acc_rate\" in feats:       parts.append(0.20 * (1 - feats[\"A4_acc_rate\"].fillna(0)))\n",
        "    if \"A3_valid_invalid_gap\" in feats:\n",
        "        parts.append(0.20 * feats[\"A3_valid_invalid_gap\"].fillna(0).abs())\n",
        "    if \"A1_rt_cv\" in feats: parts.append(0.20 * feats[\"A1_rt_cv\"].fillna(0))\n",
        "    if \"A2_rt_cv\" in feats: parts.append(0.10 * feats[\"A2_rt_cv\"].fillna(0))\n",
        "    if parts:\n",
        "        feats[\"RiskScore\"] = sum(parts)\n",
        "\n",
        "    # NaN/inf 정리\n",
        "    feats.replace([np.inf, -np.inf], np.nan, inplace=True)\n",
        "    return feats\n",
        "\n",
        "# -------- B 파생 --------\n",
        "def add_features_B(df: pd.DataFrame) -> pd.DataFrame:\n",
        "    feats = df.copy()\n",
        "    eps = 1e-6\n",
        "\n",
        "    # 0) Year-Month 단일축\n",
        "    if _has(feats, [\"Year\",\"Month\"]):\n",
        "        feats[\"YearMonthIndex\"] = feats[\"Year\"] * 12 + feats[\"Month\"]\n",
        "\n",
        "    # 1) 속도-정확도 트레이드오프 (B1~B5)\n",
        "    for k, acc_col, rt_col in [\n",
        "        (\"B1\", \"B1_acc_task1\", \"B1_rt_mean\"),\n",
        "        (\"B2\", \"B2_acc_task1\", \"B2_rt_mean\"),\n",
        "        (\"B3\", \"B3_acc_rate\",  \"B3_rt_mean\"),\n",
        "        (\"B4\", \"B4_acc_rate\",  \"B4_rt_mean\"),\n",
        "        (\"B5\", \"B5_acc_rate\",  \"B5_rt_mean\"),\n",
        "    ]:\n",
        "        if _has(feats, [rt_col, acc_col]):\n",
        "            feats[f\"{k}_speed_acc_tradeoff\"] = _safe_div(feats[rt_col], feats[acc_col], eps)\n",
        "\n",
        "    # 2) RT 변동계수(CV)\n",
        "    for k in [\"B1\",\"B2\",\"B3\",\"B4\",\"B5\"]:\n",
        "        m, s = f\"{k}_rt_mean\", f\"{k}_rt_std\"\n",
        "        if _has(feats, [m, s]):\n",
        "            feats[f\"{k}_rt_cv\"] = _safe_div(feats[s], feats[m], eps)\n",
        "\n",
        "    # 3) 간단 메타 리스크 스코어(휴리스틱)\n",
        "    parts = []\n",
        "    for k in [\"B4\",\"B5\"]:  # 주의집중/스트룹 유사 과제 가중\n",
        "        if _has(feats, [f\"{k}_rt_cv\"]):\n",
        "            parts.append(0.25 * feats[f\"{k}_rt_cv\"].fillna(0))\n",
        "    for k in [\"B3\",\"B4\",\"B5\"]:\n",
        "        acc = f\"{k}_acc_rate\" if k != \"B1\" and k != \"B2\" else None\n",
        "        if k in [\"B1\",\"B2\"]:\n",
        "            acc = f\"{k}_acc_task1\"\n",
        "        if acc in feats:\n",
        "            parts.append(0.25 * (1 - feats[acc].fillna(0)))\n",
        "    for k in [\"B1\",\"B2\"]:\n",
        "        tcol = f\"{k}_speed_acc_tradeoff\"\n",
        "        if tcol in feats:\n",
        "            parts.append(0.25 * feats[tcol].fillna(0))\n",
        "    if parts:\n",
        "        feats[\"RiskScore_B\"] = sum(parts)\n",
        "\n",
        "    feats.replace([np.inf, -np.inf], np.nan, inplace=True)\n",
        "    return feats"
      ],
      "metadata": {
        "id": "PmWtyDeuh6nC"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_A_features = add_features_A(train_A_features)\n",
        "train_B_features = add_features_B(train_B_features)\n",
        "\n",
        "print(\"A+feat:\", train_A_features.shape, \"B+feat:\", train_B_features.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9Dzr26Pwh62t",
        "outputId": "8a7b5216-a75b-43a5-e0a2-ae1b37e04b45"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "A+feat: (647241, 67) B+feat: (297526, 51)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "meta_A = train_meta[train_meta[\"Test\"]==\"A\"].reset_index(drop=True)\n",
        "meta_B = train_meta[train_meta[\"Test\"]==\"B\"].reset_index(drop=True)\n",
        "\n",
        "X_A, y_A = train_A_features.drop(columns=[\"Test_id\",\"Test\",\"PrimaryKey\",\"Age\",\"TestDate\"]), meta_A[\"Label\"].values\n",
        "X_B, y_B = train_B_features.drop(columns=[\"Test_id\",\"Test\",\"PrimaryKey\",\"Age\",\"TestDate\"]), meta_B[\"Label\"].values\n",
        "\n",
        "X_train_A, X_val_A, y_train_A, y_val_A = train_test_split(X_A, y_A, test_size=0.2, stratify=y_A, random_state=42)\n",
        "X_train_B, X_val_B, y_train_B, y_val_B = train_test_split(X_B, y_B, test_size=0.2, stratify=y_B, random_state=42)"
      ],
      "metadata": {
        "id": "hUG6JC_Eh-wB"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def train_and_eval(X_train, y_train, X_val, y_val, group_label):\n",
        "    model = lgb.LGBMClassifier(\n",
        "        objective=\"binary\",\n",
        "        metric=\"auc\",\n",
        "        n_estimators=3000,\n",
        "        learning_rate=0.05,\n",
        "        n_jobs=-1,\n",
        "        random_state=42,\n",
        "    )\n",
        "\n",
        "    model.fit(\n",
        "        X_train, y_train,\n",
        "        eval_set=[(X_val, y_val)],\n",
        "        eval_metric=\"auc\",\n",
        "        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]\n",
        "    )\n",
        "\n",
        "    val_pred = model.predict_proba(X_val)[:,1]\n",
        "    auc = roc_auc_score(y_val, val_pred)\n",
        "    print(f\"[{group_label}] Validation AUC: {auc:.4f}\")\n",
        "    return model"
      ],
      "metadata": {
        "id": "ksvATlo_iBqc"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model_A = train_and_eval(X_train_A, y_train_A, X_val_A, y_val_A, \"A\")\n",
        "model_B = train_and_eval(X_train_B, y_train_B, X_val_B, y_val_B, \"B\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yQP5H31HiIbM",
        "outputId": "e2e39fa4-e313-4d2a-db1d-f9d27a37dd19"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Info] Number of positive: 11754, number of negative: 506038\n",
            "[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.393189 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 9626\n",
            "[LightGBM] [Info] Number of data points in the train set: 517792, number of used features: 59\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.022700 -> initscore=-3.762418\n",
            "[LightGBM] [Info] Start training from score -3.762418\n",
            "Training until validation scores don't improve for 200 rounds\n",
            "[100]\tvalid_0's auc: 0.504672\n",
            "[200]\tvalid_0's auc: 0.508239\n",
            "[300]\tvalid_0's auc: 0.508657\n",
            "[400]\tvalid_0's auc: 0.507633\n",
            "Early stopping, best iteration is:\n",
            "[254]\tvalid_0's auc: 0.510804\n",
            "[A] Validation AUC: 0.5108\n",
            "[LightGBM] [Info] Number of positive: 10072, number of negative: 227948\n",
            "[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.044122 seconds.\n",
            "You can set `force_row_wise=true` to remove the overhead.\n",
            "And if memory is not enough, you can set `force_col_wise=true`.\n",
            "[LightGBM] [Info] Total Bins 6017\n",
            "[LightGBM] [Info] Number of data points in the train set: 238020, number of used features: 46\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.042316 -> initscore=-3.119358\n",
            "[LightGBM] [Info] Start training from score -3.119358\n",
            "Training until validation scores don't improve for 200 rounds\n",
            "[100]\tvalid_0's auc: 0.492003\n",
            "[200]\tvalid_0's auc: 0.490617\n",
            "Early stopping, best iteration is:\n",
            "[7]\tvalid_0's auc: 0.499922\n",
            "[B] Validation AUC: 0.4999\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import joblib\n",
        "\n",
        "trial = 1\n",
        "\n",
        "# 모델 저장 경로\n",
        "os.makedirs(\"/content/drive/MyDrive/dacon_driver/model\", exist_ok=True)\n",
        "\n",
        "joblib.dump(model_A, f\"/content/drive/MyDrive/dacon_driver/pkl/A_{trial}.pkl\")\n",
        "joblib.dump(model_B, f\"/content/drive/MyDrive/dacon_driver/pkl/B_{trial}.pkl\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dZfQ5-e0iLJp",
        "outputId": "17b8f67a-edfb-4482-a66a-ac8508a54ca2"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['/content/drive/MyDrive/dacon_driver/pkl/B_1.pkl']"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "tvjacfnV_Mpl"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}