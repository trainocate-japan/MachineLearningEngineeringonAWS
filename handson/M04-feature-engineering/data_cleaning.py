"""
モジュール 4: データクリーニング

M03 で特定したデータ品質問題を修正します:
- 書式ゆれ（income_text）→ 数値化
- 欠損値（age, state）→ 補完
- 重複行 → 排除
- 外れ値（age>=120）→ 処理

扱う内容:
- 不正確・重複・欠落したデータの処理

出力: ../common/data/cleaned_income.csv （M04 の後続スクリプトが使用）
"""

import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_income.csv")
OUT_CSV = os.path.join(DATA_DIR, "cleaned_income.csv")


def parse_income(text) -> float:
    """
    書式ゆれのある収入文字列を数値へ変換する。
    例: "73K" -> 73000, "$93605" -> 93605, "117,451" -> 117451
    """
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
    report = {}

    # 1. 書式ゆれ: income_text -> income_usd（数値）
    df["income_usd"] = df["income_text"].apply(parse_income)
    df = df.drop(columns=["income_text"])

    # 2. 重複行の排除
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["removed_duplicates"] = before - len(df)

    # 3. 外れ値: ありえない年齢（>=120）を欠損扱いにして後で補完
    out_mask = df["age"] >= 120
    report["age_outliers"] = int(out_mask.sum())
    df.loc[out_mask, "age"] = pd.NA

    # 4. 欠損値の補完
    #    - age: 中央値で補完（外れ値の影響を受けにくい）
    #    - income_usd: 中央値で補完
    #    - state: 最頻値で補完
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median).astype(float)

    income_median = df["income_usd"].median()
    report["income_filled"] = int(df["income_usd"].isna().sum())
    df["income_usd"] = df["income_usd"].fillna(income_median)

    state_mode = df["state"].mode().iloc[0]
    report["state_filled"] = int(df["state"].isna().sum())
    df["state"] = df["state"].fillna(state_mode)

    return df, report


def main():
    print("=" * 60)
    print(" データクリーニング")
    print("=" * 60)

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {RAW_CSV}\n"
            "先に以下を実行してください: python ../common/generate_dataset.py"
        )

    df = pd.read_csv(RAW_CSV)
    print(f"\n  入力: {RAW_CSV}")
    print(f"  入力行数: {len(df)}, 欠損セル: {int(df.isna().sum().sum())}")

    cleaned, report = clean(df)

    print("\n  [クリーニング結果]")
    print(f"  - 重複排除: {report['removed_duplicates']} 行削除")
    print(f"  - 年齢の外れ値(>=120): {report['age_outliers']} 件を補完対象化")
    print(f"  - age 補完後の欠損: {int(cleaned['age'].isna().sum())}")
    print(f"  - income_usd 補完: {report['income_filled']} 件")
    print(f"  - state 補完: {report['state_filled']} 件")
    print(f"\n  出力行数: {len(cleaned)}, 残存欠損セル: {int(cleaned.isna().sum().sum())}")

    cleaned.to_csv(OUT_CSV, index=False)
    print(f"\n  クリーン済みデータを保存: {OUT_CSV}")
    print("\n  [先頭 5 行]")
    print(cleaned.head().to_string(index=False))
    print("\n  ✅ データクリーニング完了")


if __name__ == "__main__":
    main()
