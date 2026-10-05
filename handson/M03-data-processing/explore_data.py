"""
モジュール 3: 探索的データ分析（EDA）

pandas を使用して「成人の収入予測」生データセットを探索し、
データ品質問題（欠損・重複・外れ値・書式ゆれ）を特定します。

扱う内容:
- 探索的データ分析（EDA）
- 不正確・重複・欠落したデータの検出
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_income.csv")


def load_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {RAW_CSV}\n"
            "先に以下を実行してください: python ../common/generate_dataset.py"
        )
    return pd.read_csv(RAW_CSV)


def section(title: str):
    print("\n" + "─" * 60)
    print(f"  {title}")
    print("─" * 60)


def run_eda():
    print("=" * 60)
    print(" 探索的データ分析（EDA）: 成人の収入予測データセット")
    print("=" * 60)

    df = load_data()

    # -------------------------------------------------------------------------
    # 1. 基本情報
    # -------------------------------------------------------------------------
    section("1. 基本情報（shape / dtypes）")
    print(f"  行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print("\n  データ型:")
    print(df.dtypes.to_string())

    # -------------------------------------------------------------------------
    # 2. 記述統計
    # -------------------------------------------------------------------------
    section("2. 数値列の記述統計")
    print(df.describe().round(2).to_string())
    print("\n  [観察] age の max を確認してください。外れ値の可能性があります。")

    # -------------------------------------------------------------------------
    # 3. 欠損値の確認
    # -------------------------------------------------------------------------
    section("3. 欠損値（列ごと）")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        print("  欠損値なし")
    else:
        for col, cnt in miss.items():
            print(f"  {col}: {cnt} 件 ({cnt / len(df) * 100:.1f}%)")

    # -------------------------------------------------------------------------
    # 4. 重複行の確認
    # -------------------------------------------------------------------------
    section("4. 重複行")
    dup_count = df.duplicated().sum()
    print(f"  完全重複行: {dup_count} 件")

    # -------------------------------------------------------------------------
    # 5. 外れ値の確認（IQR 法）
    # -------------------------------------------------------------------------
    section("5. 外れ値の確認（age, IQR 法）")
    age = df["age"].dropna()
    q1, q3 = age.quantile(0.25), age.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    outliers = age[(age > upper) | (age < lower)]
    print(f"  Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
    print(f"  許容範囲: [{lower:.1f}, {upper:.1f}]")
    print(f"  外れ値の件数: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  外れ値の例: {sorted(outliers.unique())[:10]}")

    # -------------------------------------------------------------------------
    # 6. 書式ゆれの確認（income_text）
    # -------------------------------------------------------------------------
    section("6. 書式ゆれ（income_text）")
    print("  income は数値であるべきですが、文字列として混在した書式になっています:")
    print(df["income_text"].head(8).to_string(index=False))
    print("\n  [必要な処理] 'K' 展開、'$' と ',' の除去 → 数値化（M04 で実施）")

    # -------------------------------------------------------------------------
    # 7. ターゲットの分布
    # -------------------------------------------------------------------------
    section("7. ターゲット（income_high）のクラスバランス")
    vc = df["income_high"].value_counts().sort_index()
    for label, cnt in vc.items():
        name = "高収入(1)" if label == 1 else "低収入(0)"
        print(f"  {name}: {cnt} 件 ({cnt / len(df) * 100:.1f}%)")

    # -------------------------------------------------------------------------
    # まとめ
    # -------------------------------------------------------------------------
    section("まとめ: 特定されたデータ品質問題")
    print("  ① 欠損値 (age, state) → 補完 or 行削除")
    print("  ② 重複行 → 重複排除")
    print("  ③ 外れ値 (age>=120) → 削除 or 上限クリップ")
    print("  ④ 書式ゆれ (income_text) → パースして数値化")
    print("\n  これらは M04（データ変換と特徴量エンジニアリング）で対処します。")
    print("\n  ✅ EDA 完了")


if __name__ == "__main__":
    run_eda()
