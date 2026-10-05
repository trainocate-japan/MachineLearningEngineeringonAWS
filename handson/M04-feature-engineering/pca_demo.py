"""
モジュール 4: PCA による次元削減

扱う内容:
- 特徴量選択テクニック
  - 主成分分析（PCA）: 複数の特徴量を主成分に変換し、分散でランク付け

住宅データ（面積/寝室/バスルームは「サイズ」という主成分に集約できる）と同じ原理の
デモです。ここでは収入予測データの数値特徴量に PCA を適用し、少数の主成分で
分散の大部分を説明できることを示します。
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
IN_CSV = os.path.join(DATA_DIR, "cleaned_income.csv")

NUMERIC_FEATURES = ["age", "education_num", "hours_per_week", "income_usd"]


def main():
    print("=" * 60)
    print(" PCA による次元削減デモ")
    print("=" * 60)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {IN_CSV}\n"
            "先に以下を実行してください: python data_cleaning.py"
        )

    df = pd.read_csv(IN_CSV)
    X = df[NUMERIC_FEATURES].values

    # PCA の前に標準化（スケールの異なる特徴量を揃える）
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=len(NUMERIC_FEATURES))
    pca.fit(X_std)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    print(f"\n  入力特徴量 ({len(NUMERIC_FEATURES)} 次元): {NUMERIC_FEATURES}")
    print("\n  [各主成分の説明分散比]")
    for i, (r, c) in enumerate(zip(evr, cum), 1):
        bar = "█" * int(r * 40)
        print(f"  PC{i}: {r * 100:5.1f}%  (累積 {c * 100:5.1f}%)  {bar}")

    # 分散の 90% を説明するのに必要な主成分数
    n_90 = int(np.argmax(cum >= 0.90) + 1)
    print(f"\n  分散の 90% を説明するのに必要な主成分数: {n_90} / {len(NUMERIC_FEATURES)}")

    print("\n  [第1主成分の負荷量（各特徴量の寄与）]")
    loadings = pca.components_[0]
    for feat, load in sorted(zip(NUMERIC_FEATURES, loadings), key=lambda t: -abs(t[1])):
        print(f"    {feat:16s}: {load:+.3f}")

    print("\n  [考察ポイント]")
    print("  - 相関の高い特徴量は少数の主成分に集約される")
    print("  - 次元削減により過学習の抑制・学習の高速化・解釈性向上が期待できる")
    print("  - ただし主成分は元の特徴量の線形結合のため、解釈性とのトレードオフに注意")
    print("\n  ✅ PCA デモ完了")


if __name__ == "__main__":
    main()
