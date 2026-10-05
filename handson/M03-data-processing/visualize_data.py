"""
モジュール 3: データ可視化

目的別に可視化手法を使い分けます:
- 分布: ヒストグラム / ボックスプロット
- 関係: 散布図 / 相関ヒートマップ
- 構成: 棒グラフ

生成した図は output/ に PNG として保存します（ヘッドレス環境対応）。
--publish を付けると S3 にアップロードし、ブラウザ閲覧用の署名付き URL を発行します。

扱う内容:
- 探索的データ分析における可視化（データカテゴリの可視化 / 数値データの可視化）
"""

import os
import sys
import argparse
import matplotlib

matplotlib.use("Agg")  # ヘッドレス（EC2）環境で描画するためのバックエンド
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK", "IPAexGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_income.csv")
OUT_DIR = os.path.join(HERE, "output")


def load_data() -> pd.DataFrame:
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {CLEAN_CSV}\n"
            "先に以下を実行してください: python ../common/generate_dataset.py"
        )
    return pd.read_csv(CLEAN_CSV)


def run_visualizations():
    print("=" * 60)
    print(" データ可視化: 成人の収入予測データセット")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    saved = []

    # -------------------------------------------------------------------------
    # 1. 分布: ヒストグラム（age）
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["age"], bins=30, color="#4C78A8", edgecolor="white")
    ax.set_title("年齢の分布 (ヒストグラム = 分布)")
    ax.set_xlabel("age")
    ax.set_ylabel("count")
    p = os.path.join(OUT_DIR, "hist_age.png")
    fig.tight_layout()
    fig.savefig(p, dpi=100)
    plt.close(fig)
    saved.append(p)

    # -------------------------------------------------------------------------
    # 2. 分布: ボックスプロット（income_usd）
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(df["income_usd"], vert=False, widths=0.6)
    ax.set_title("収入のボックスプロット (外れ値の確認)")
    ax.set_xlabel("income_usd")
    p = os.path.join(OUT_DIR, "box_income.png")
    fig.tight_layout()
    fig.savefig(p, dpi=100)
    plt.close(fig)
    saved.append(p)

    # -------------------------------------------------------------------------
    # 3. 関係: 散布図（education_num vs income_usd）
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = df["income_high"].map({0: "#B0B0B0", 1: "#E45756"})
    ax.scatter(df["education_num"], df["income_usd"], c=colors, s=10, alpha=0.5)
    ax.set_title("教育年数 vs 収入 (散布図 = 関係)")
    ax.set_xlabel("education_num")
    ax.set_ylabel("income_usd")
    p = os.path.join(OUT_DIR, "scatter_edu_income.png")
    fig.tight_layout()
    fig.savefig(p, dpi=100)
    plt.close(fig)
    saved.append(p)

    # -------------------------------------------------------------------------
    # 4. 関係: 相関ヒートマップ（数値列）
    # -------------------------------------------------------------------------
    num_cols = ["age", "education_num", "hours_per_week", "income_usd", "income_high"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("相関ヒートマップ (関係)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = os.path.join(OUT_DIR, "heatmap_corr.png")
    fig.tight_layout()
    fig.savefig(p, dpi=100)
    plt.close(fig)
    saved.append(p)

    # -------------------------------------------------------------------------
    # 5. 構成: 棒グラフ（学歴別の高収入割合）
    # -------------------------------------------------------------------------
    grp = df.groupby("education")["income_high"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(grp.index, grp.values, color="#54A24B")
    ax.set_title("学歴別の高収入割合 (棒グラフ = 比較/構成)")
    ax.set_xlabel("高収入(income_high=1)の割合")
    p = os.path.join(OUT_DIR, "bar_education.png")
    fig.tight_layout()
    fig.savefig(p, dpi=100)
    plt.close(fig)
    saved.append(p)

    print("\n  保存した図:")
    for s in saved:
        print(f"  - {s}")

    print("\n  [考察ポイント]")
    print("  - education_num と income_usd / income_high に正の相関が見える")
    print("  - income のボックスプロットで右側に外れ値（高収入層）が存在")
    print("  - 可視化の目的（分布 / 関係 / 構成）でグラフ種別を選ぶ")
    print("\n  ✅ 可視化完了")
    return saved


def _publish(paths):
    """生成した画像を S3 にアップし署名付き URL を表示する。"""
    sys.path.insert(0, os.path.join(HERE, ".."))
    from common.publish import publish_file

    print("\n  [ブラウザ閲覧用の署名付き URL]")
    for p in paths:
        try:
            print(f"  - {os.path.basename(p)}: {publish_file(p)}")
        except Exception as e:
            print(f"  ⚠️  {p}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true",
                        help="生成画像を S3 にアップしブラウザ閲覧用 URL を発行")
    args = parser.parse_args()
    saved = run_visualizations()
    if args.publish:
        _publish(saved)
