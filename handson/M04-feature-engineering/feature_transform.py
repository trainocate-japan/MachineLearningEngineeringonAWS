"""
モジュール 4: 特徴量変換（エンコーディング・スケーリング・ビニング）

扱う内容:
- 特徴量エンジニアリングの概念
  - カテゴリカルエンコーディング（ラベル / 序数 / ワンホット）
  - 数値特徴量エンジニアリング（正規化 / 標準化 / ビニング / 対数変換）

入力: ../common/data/cleaned_income.csv （data_cleaning.py の出力）
出力: ../common/data/features_income.csv （M06 のトレーニング用）
      先頭列を income_high（ターゲット）にし、XGBoost 等が扱いやすい形にする
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
IN_CSV = os.path.join(DATA_DIR, "cleaned_income.csv")
OUT_CSV = os.path.join(DATA_DIR, "features_income.csv")

# 序数（順序あり）カテゴリ: education は学歴の高低に順序がある
EDUCATION_ORDER = {"高校卒業証書": 0, "専門学校": 1, "4年制大学卒業": 2, "修士/博士": 3}


def transform(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    # -------------------------------------------------------------------------
    # ターゲット
    # -------------------------------------------------------------------------
    out["income_high"] = df["income_high"].astype(int)

    # -------------------------------------------------------------------------
    # カテゴリカルエンコーディング
    # -------------------------------------------------------------------------
    # (1) 序数エンコーディング: education（順序あり）
    out["education_ordinal"] = df["education"].map(EDUCATION_ORDER).astype(int)

    # (2) ワンホットエンコーディング: workclass, occupation, state（順序なし名義変数）
    for col in ["workclass", "occupation", "state"]:
        dummies = pd.get_dummies(df[col], prefix=col).astype(int)
        out = pd.concat([out, dummies], axis=1)

    # -------------------------------------------------------------------------
    # 数値特徴量エンジニアリング
    # -------------------------------------------------------------------------
    # (3) 標準化: age, hours_per_week（平均0・標準偏差1、外れ値の影響を軽減）
    std = StandardScaler()
    out[["age_std", "hours_std"]] = std.fit_transform(df[["age", "hours_per_week"]])

    # (4) 正規化: education_num（0〜1 スケール）
    mm = MinMaxScaler()
    out[["education_num_norm"]] = mm.fit_transform(df[["education_num"]])

    # (5) 対数変換: income_usd（右に歪んだ分布を圧縮）
    out["income_log"] = np.log1p(df["income_usd"])

    # (6) ビニング: age を世代カテゴリに分割し、さらに序数化
    bins = [0, 30, 45, 60, 200]
    labels = [0, 1, 2, 3]  # 若年 / 中堅 / 熟練 / シニア
    out["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(int)

    return out


def main():
    print("=" * 60)
    print(" 特徴量変換（エンコーディング・スケーリング・ビニング）")
    print("=" * 60)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {IN_CSV}\n"
            "先に以下を実行してください: python data_cleaning.py"
        )

    df = pd.read_csv(IN_CSV)
    print(f"\n  入力: {IN_CSV}（{df.shape[0]} 行 / {df.shape[1]} 列）")

    feats = transform(df)

    print("\n  [適用した変換]")
    print("  - 序数エンコーディング: education -> education_ordinal")
    print("  - ワンホットエンコーディング: workclass / occupation / state")
    print("  - 標準化: age -> age_std, hours_per_week -> hours_std")
    print("  - 正規化: education_num -> education_num_norm")
    print("  - 対数変換: income_usd -> income_log")
    print("  - ビニング: age -> age_bin（世代 0..3）")

    print(f"\n  出力特徴量: {feats.shape[1]} 列（先頭が income_high=ターゲット）")
    print("  列名:")
    print("   ", ", ".join(feats.columns.tolist()))

    feats.to_csv(OUT_CSV, index=False)
    print(f"\n  特徴量データを保存: {OUT_CSV}")
    print("\n  [先頭 3 行]")
    print(feats.head(3).to_string(index=False))
    print("\n  ✅ 特徴量変換完了")


if __name__ == "__main__":
    main()
