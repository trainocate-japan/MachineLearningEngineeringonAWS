"""
モジュール 7: バイアスと分散のトレードオフ

モデルの複雑さ（XGBoost の max_depth）を変化させ、
過小適合（高バイアス）→ 理想 → 過学習（高分散）の遷移を観察します。

扱う内容:
- モデルのバイアスと分散のトレードオフ
  （過小適合による高バイアス / 過学習による高分散）

入力: ../common/data/features_income.csv
出力: output/bias_variance.png
"""

import os
import sys
import argparse
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK", "IPAexGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
IN_CSV = os.path.join(DATA_DIR, "features_income.csv")
OUT_DIR = os.path.join(HERE, "output")


def main():
    print("=" * 60)
    print(" バイアスと分散のトレードオフ")
    print("=" * 60)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {IN_CSV}\n"
            "先に M04 の feature_transform.py を実行してください。"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    y = df["income_high"].values
    X = df.drop(columns=["income_high"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # モデルの複雑さを変化させる（max_depth と num_round を同時に増やす）
    configs = [
        (1, 10, "過小適合寄り"),
        (2, 30, ""),
        (4, 80, "バランス"),
        (8, 300, ""),
        (12, 800, "過学習寄り"),
    ]

    train_losses, test_losses, labels = [], [], []
    print("\n  複雑さ別の損失（logloss, 低いほど良い）:")
    print("  depth  rounds  train    test     ギャップ")
    for depth, rounds, note in configs:
        params = {"objective": "binary:logistic", "max_depth": depth, "eta": 0.2}
        model = xgb.train(params, dtrain, num_boost_round=rounds)
        tr = log_loss(y_train, model.predict(dtrain))
        te = log_loss(y_test, model.predict(dtest))
        train_losses.append(tr)
        test_losses.append(te)
        labels.append(f"d{depth}/r{rounds}")
        gap = te - tr
        note_s = f"  ← {note}" if note else ""
        print(f"   {depth:<5}  {rounds:<6}  {tr:.4f}  {te:.4f}  {gap:+.4f}{note_s}")

    # 可視化
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(configs))
    ax.plot(x, train_losses, "o-", label="train loss（バイアスの目安）")
    ax.plot(x, test_losses, "s-", label="test loss（汎化性能）")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("モデルの複雑さ（低 → 高）")
    ax.set_ylabel("logloss")
    ax.set_title("バイアス・分散のトレードオフ")
    ax.legend()
    p = os.path.join(OUT_DIR, "bias_variance.png")
    fig.tight_layout(); fig.savefig(p, dpi=100); plt.close(fig)

    print(f"\n  図を保存: {p}")
    print("\n  [観察ポイント]")
    print("  - 左側（単純なモデル）: train/test ともに損失が高い = 高バイアス（過小適合）")
    print("  - 右側（複雑なモデル）: train は下がるが test が下げ止まり/上昇 = 高分散（過学習）")
    print("  - 中央付近が理想: test loss が最小になる複雑さを選ぶ")
    print("\n  → 理想は『バイアス低・分散低』。これをチューニングで探します（次の AMT デモ）。")
    print("\n  ✅ バイアス・分散デモ完了")
    return p


def _publish(paths):
    sys.path.insert(0, os.path.join(HERE, ".."))
    from common.publish import publish_file

    print("\n  [ブラウザ閲覧用の署名付き URL]")
    for pth in paths:
        try:
            print(f"  - {os.path.basename(pth)}: {publish_file(pth)}")
        except Exception as e:
            print(f"  ⚠️  {pth}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true",
                        help="バイアス分散の図を S3 にアップし URL を発行")
    args = parser.parse_args()
    png = main()
    if args.publish and png:
        _publish([png])
