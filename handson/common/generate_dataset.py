"""
共通サンプルデータセット生成スクリプト

Machine Learning Engineering on AWS ハンズオン全体で使用する
「成人の収入予測」データセットを生成します。
（UCI Adult Income データセットを模した合成データ）

ターゲット: income_high (収入が閾値 50K を超えるか = 二項分類)

生成されるファイル:
- data/raw_income.csv   : 意図的にデータ品質問題を含む生データ（M03/M04 用）
- data/clean_income.csv : クリーン済みデータ（M06 以降のトレーニング用）

想定するタスクは、人口統計から収入が閾値を超えるかを予測する二項分類です。
"""

import os
import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# =============================================================================
# カテゴリ定義
# =============================================================================
EDUCATION_LEVELS = ["高校卒業証書", "4年制大学卒業", "修士/博士", "専門学校"]
EDUCATION_YEARS = {"高校卒業証書": 12, "専門学校": 14, "4年制大学卒業": 16, "修士/博士": 20}
WORKCLASS = ["民間企業", "公務員", "自営業", "非営利団体"]
STATES = ["NY", "CA", "TX", "FL", "IL", "WA", "NC", "MN"]
OCCUPATIONS = ["エンジニア", "営業", "管理職", "事務", "サービス", "専門職"]


def _base_frame(n: int) -> pd.DataFrame:
    """収入と相関を持つ特徴量で基本データフレームを作る"""
    age = RNG.integers(18, 70, size=n)
    education = RNG.choice(EDUCATION_LEVELS, size=n, p=[0.35, 0.35, 0.15, 0.15])
    education_num = np.array([EDUCATION_YEARS[e] for e in education])
    hours_per_week = RNG.normal(40, 10, size=n).clip(5, 80).round().astype(int)
    workclass = RNG.choice(WORKCLASS, size=n, p=[0.6, 0.2, 0.15, 0.05])
    state = RNG.choice(STATES, size=n)
    occupation = RNG.choice(OCCUPATIONS, size=n)

    # 収入は年齢・教育年数・労働時間に相関する潜在スコアから生成
    latent = (
        0.03 * (age - 18)
        + 0.18 * (education_num - 12)
        + 0.04 * (hours_per_week - 40)
        + RNG.normal(0, 1.0, size=n)
    )
    prob_high = 1 / (1 + np.exp(-latent + 1.5))
    income_high = (RNG.random(n) < prob_high).astype(int)

    # 実額の年収（ドル）も付与（EDA・外れ値デモ用）
    base_income = 25000 + education_num * 2200 + age * 400 + hours_per_week * 300
    income_usd = (base_income + income_high * 25000 + RNG.normal(0, 6000, size=n)).clip(8000, None)

    return pd.DataFrame(
        {
            "age": age,
            "workclass": workclass,
            "education": education,
            "education_num": education_num,
            "occupation": occupation,
            "hours_per_week": hours_per_week,
            "state": state,
            "income_usd": income_usd.round().astype(int),
            "income_high": income_high,
        }
    )


def make_clean(n: int) -> pd.DataFrame:
    """トレーニング用のクリーンなデータセット"""
    return _base_frame(n)


def make_raw(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    EDA / データクリーニングのデモ用に、意図的に品質問題を注入した生データ。
    注入する問題:
      - 欠損値（age, income_usd, state）
      - 重複行
      - 外れ値（age=154 など）
      - 書式ゆれ（income を文字列で "45K", "$37000", "42,000" 等）
    """
    df = df_clean.copy()
    n = len(df)

    # income_usd を書式ゆれのある文字列列 income_text に変換
    def messy_income(v, i):
        if i % 11 == 0:
            return f"{v // 1000}K"
        if i % 7 == 0:
            return f"${v}"
        if i % 5 == 0:
            return f"{v:,}"
        return str(v)

    df["income_text"] = [messy_income(v, i) for i, v in enumerate(df["income_usd"])]
    df = df.drop(columns=["income_usd"])

    # 欠損値を注入
    miss_age = RNG.choice(n, size=max(1, n // 25), replace=False)
    df.loc[miss_age, "age"] = np.nan
    miss_state = RNG.choice(n, size=max(1, n // 20), replace=False)
    df.loc[miss_state, "state"] = np.nan

    # 外れ値を注入（ありえない年齢）
    out_idx = RNG.choice(n, size=max(1, n // 100), replace=False)
    df.loc[out_idx, "age"] = RNG.integers(120, 160, size=len(out_idx))

    # 重複行を追加
    dup = df.sample(n=max(1, n // 50), random_state=1)
    df = pd.concat([df, dup], ignore_index=True)

    # 列順を整える
    cols = ["age", "workclass", "education", "education_num",
            "occupation", "hours_per_week", "state", "income_text", "income_high"]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="ハンズオン共通データセット生成")
    parser.add_argument("--rows", type=int, default=4000, help="生成する行数")
    parser.add_argument("--out-dir", default=None, help="出力ディレクトリ（既定: このスクリプトと同階層の data/）")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(" ハンズオン共通データセット生成")
    print("=" * 60)

    df_clean = make_clean(args.rows)
    df_raw = make_raw(df_clean)

    clean_path = os.path.join(out_dir, "clean_income.csv")
    raw_path = os.path.join(out_dir, "raw_income.csv")
    df_clean.to_csv(clean_path, index=False)
    df_raw.to_csv(raw_path, index=False)

    print(f"\n  クリーンデータ: {clean_path}")
    print(f"    行数: {len(df_clean)}, 列数: {df_clean.shape[1]}")
    print(f"    income_high 比率: {df_clean['income_high'].mean():.3f}")
    print(f"\n  生データ（品質問題あり）: {raw_path}")
    print(f"    行数: {len(df_raw)}（重複含む）, 列数: {df_raw.shape[1]}")
    print(f"    欠損セル数: {int(df_raw.isna().sum().sum())}")

    print("\n  [先頭 5 行（生データ）]")
    print(df_raw.head().to_string(index=False))
    print("\n  ✅ データセット生成完了")


if __name__ == "__main__":
    main()
