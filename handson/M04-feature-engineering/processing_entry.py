"""
SageMaker Processing のエントリースクリプト

このスクリプトは SageMaker Processing コンテナ内で実行されます。
入力の生データをクリーニング・特徴量変換し、train/validation/test に分割して出力します。

ローカルの data_cleaning.py / feature_transform.py と同じロジックを
1 ファイルに統合しています（コンテナ内で外部モジュールに依存しないため）。

- 入力:  /opt/ml/processing/input/raw_income.csv
- 出力:  /opt/ml/processing/output/{train,validation,test}/*.csv
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

EDUCATION_ORDER = {"高校卒業証書": 0, "専門学校": 1, "4年制大学卒業": 2, "修士/博士": 3}


def parse_income(text) -> float:
    if pd.isna(text):
        return float("nan")
    s = str(text).strip().replace("$", "").replace(",", "")
    m = re.match(r"^(\d+(?:\.\d+)?)([Kk])$", s)
    if m:
        return float(m.group(1)) * 1000
    try:
        return float(s)
    except ValueError:
        return float("nan")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["income_usd"] = df["income_text"].apply(parse_income)
    df = df.drop(columns=["income_text"])
    df = df.drop_duplicates().reset_index(drop=True)
    df.loc[df["age"] >= 120, "age"] = pd.NA
    df["age"] = df["age"].fillna(df["age"].median()).astype(float)
    df["income_usd"] = df["income_usd"].fillna(df["income_usd"].median())
    df["state"] = df["state"].fillna(df["state"].mode().iloc[0])
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["income_high"] = df["income_high"].astype(int)
    out["education_ordinal"] = df["education"].map(EDUCATION_ORDER).astype(int)
    for col in ["workclass", "occupation", "state"]:
        out = pd.concat([out, pd.get_dummies(df[col], prefix=col).astype(int)], axis=1)
    std = StandardScaler()
    out[["age_std", "hours_std"]] = std.fit_transform(df[["age", "hours_per_week"]])
    out[["education_num_norm"]] = MinMaxScaler().fit_transform(df[["education_num"]])
    out["income_log"] = np.log1p(df["income_usd"])
    bins = [0, 30, 45, 60, 200]
    out["age_bin"] = pd.cut(df["age"], bins=bins, labels=[0, 1, 2, 3], right=False).astype(int)
    return out


def main():
    print("[processing] 入力データを読み込み中...")
    raw_path = os.path.join(INPUT_DIR, "raw_income.csv")
    df = pd.read_csv(raw_path)
    print(f"[processing] 入力行数: {len(df)}")

    cleaned = clean(df)
    feats = transform(cleaned)
    print(f"[processing] 特徴量: {feats.shape[0]} 行 / {feats.shape[1]} 列")

    # train 70% / validation 15% / test 15% に層化分割
    train, temp = train_test_split(
        feats, test_size=0.30, random_state=42, stratify=feats["income_high"]
    )
    valid, test = train_test_split(
        temp, test_size=0.50, random_state=42, stratify=temp["income_high"]
    )

    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        out_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        # XGBoost 向けにヘッダなし・先頭列ターゲットで保存
        part.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, header=False)
        print(f"[processing] {name}: {len(part)} 行 -> {out_dir}")

    print("[processing] 完了")


if __name__ == "__main__":
    main()
