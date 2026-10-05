"""
モジュール 7: モデル評価

混同行列・Accuracy・Precision・Recall・F1・ROC/AUC で
二項分類モデルを評価します。

扱う内容:
- モデルパフォーマンスの評価
  （混同行列 / 精度 / 再現率・適合率 / F1 スコア / ROC 曲線・AUC）

入力: ../common/data/features_income.csv
出力: output/roc_curve.png, output/confusion_matrix.png
"""

import os
import sys
import argparse
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK", "IPAexGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
IN_CSV = os.path.join(DATA_DIR, "features_income.csv")
OUT_DIR = os.path.join(HERE, "output")


def train_quick(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.2}
    return xgb.train(params, dtrain, num_boost_round=80)


def main():
    print("=" * 60)
    print(" モデル評価: 分類指標")
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_quick(X_train, y_train)
    prob = model.predict(xgb.DMatrix(X_test))
    pred = (prob >= 0.5).astype(int)

    # -------------------------------------------------------------------------
    # 混同行列
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n  [混同行列]")
    print(f"                 予測:低(0)   予測:高(1)")
    print(f"    実際:低(0)      TN={tn:<6}   FP={fp:<6}")
    print(f"    実際:高(1)      FN={fn:<6}   TP={tp:<6}")

    # -------------------------------------------------------------------------
    # 各指標
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print("\n  [評価指標]")
    print(f"    Accuracy  (正解率):          {acc:.4f}")
    print(f"    Precision (適合率 TP/(TP+FP)): {prec:.4f}")
    print(f"    Recall    (再現率 TP/(TP+FN)): {rec:.4f}")
    print(f"    F1 Score  (調和平均):         {f1:.4f}")
    print(f"    AUC       (ROC 曲線下面積):    {auc:.4f}")

    print("\n  [指標の使い分け]")
    print("    - Precision 重視: 誤検知(FP)のコストが高い（例: スパム判定で正常メールを誤判定したくない）")
    print("    - Recall 重視:    見逃し(FN)のコストが高い（例: 疾病検出・不正検出）")
    print("    - F1:             Precision と Recall のバランス")
    print("    - AUC:            閾値に依存しない総合的な分離性能（0.5=ランダム, 1.0=完璧）")

    # -------------------------------------------------------------------------
    # 混同行列ヒートマップ
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["低(0)", "高(1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["低(0)", "高(1)"])
    ax.set_xlabel("予測"); ax.set_ylabel("実際")
    ax.set_title("混同行列")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.tight_layout(); fig.savefig(cm_path, dpi=100); plt.close(fig)

    # -------------------------------------------------------------------------
    # ROC 曲線
    # -------------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", color="#E45756")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ランダム (AUC=0.5)")
    ax.set_xlabel("偽陽性率 (FPR)")
    ax.set_ylabel("真陽性率 (TPR)")
    ax.set_title("ROC 曲線")
    ax.legend(loc="lower right")
    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    fig.tight_layout(); fig.savefig(roc_path, dpi=100); plt.close(fig)

    print(f"\n  混同行列: {cm_path}")
    print(f"  ROC 曲線: {roc_path}")
    print("\n  ✅ モデル評価完了")
    return [cm_path, roc_path]


def _publish(paths):
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
                        help="混同行列・ROC 曲線の PNG を S3 にアップし URL を発行")
    args = parser.parse_args()
    pngs = main()
    if args.publish and pngs:
        _publish(pngs)
