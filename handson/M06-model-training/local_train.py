"""
モジュール 6: ローカル XGBoost トレーニング

SageMaker に投げる前に、ローカルで XGBoost を学習させてモデルの挙動を理解します。
学習曲線（トレーニング損失 vs 検証損失）から、フィッティング状態を観察します。

扱う内容:
- モデルトレーニングの基本概念
  （モデル=アルゴリズム+重み / 損失関数 / 勾配降下法 / ハイパーパラメータ /
    過学習・過小適合・理想的フィッティング）

入力: ../common/data/features_income.csv （M04 feature_transform.py の出力）
出力: output/model_local.json, output/learning_curve.png
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
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
IN_CSV = os.path.join(DATA_DIR, "features_income.csv")
OUT_DIR = os.path.join(HERE, "output")


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


def main():
    print("=" * 60)
    print(" ローカル XGBoost トレーニング")
    print("=" * 60)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {IN_CSV}\n"
            "先に以下を実行してください（M04）:\n"
            "  python ../common/generate_dataset.py\n"
            "  python ../M04-feature-engineering/data_cleaning.py\n"
            "  python ../M04-feature-engineering/feature_transform.py"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)

    # 先頭列がターゲット income_high
    y = df["income_high"].values
    X = df.drop(columns=["income_high"]).values
    feature_names = df.drop(columns=["income_high"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # XGBoost のハイパーパラメータ
    params = {
        "objective": "binary:logistic",   # 学習タスク: 二項分類
        "eval_metric": ["logloss", "auc"], # 評価メトリクス
        "max_depth": 4,                    # ツリーの最大深度
        "eta": 0.2,                        # 学習率
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    num_round = 100  # ラウンド数（ツリー数）

    print("\n  [ハイパーパラメータ]")
    for k, v in params.items():
        print(f"    {k}: {v}")
    print(f"    num_round: {num_round}")

    # 学習（トレーニング損失と検証損失を記録）
    evals_result: dict = {}
    print("\n  トレーニング中...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dtest, "validation")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    # 評価
    pred_prob = model.predict(dtest)
    pred = (pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_prob)

    print("\n  [テストセット評価]")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    AUC:      {auc:.4f}")

    # 学習曲線の保存
    train_ll = evals_result["train"]["logloss"]
    valid_ll = evals_result["validation"]["logloss"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_ll, label="train logloss")
    ax.plot(valid_ll, label="validation logloss")
    ax.set_title("学習曲線（損失の推移でフィッティングを観察）")
    ax.set_xlabel("boosting round")
    ax.set_ylabel("logloss")
    ax.legend()
    curve_path = os.path.join(OUT_DIR, "learning_curve.png")
    fig.tight_layout()
    fig.savefig(curve_path, dpi=100)
    plt.close(fig)

    # 特徴量重要度（上位）
    importance = model.get_score(importance_type="gain")
    top = sorted(importance.items(), key=lambda t: -t[1])[:8]
    print("\n  [特徴量重要度 上位8（gain）]")
    for feat, score in top:
        print(f"    {feat:22s}: {score:.1f}")

    # モデル保存
    model_path = os.path.join(OUT_DIR, "model_local.json")
    model.save_model(model_path)

    print(f"\n  学習曲線: {curve_path}")
    print(f"  モデル保存: {model_path}")

    print("\n  [考察ポイント]")
    print("  - train と validation の logloss が離れていくと過学習の兆候")
    print("  - 両方高止まりなら過小適合（モデルが単純すぎ / 特徴量不足）")
    print("  - 理想は両者が低く、かつ乖離が小さい状態")
    print("\n  ✅ ローカルトレーニング完了")
    return curve_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true",
                        help="学習曲線 PNG を S3 にアップしブラウザ閲覧用 URL を発行")
    args = parser.parse_args()
    curve = main()
    if args.publish and curve:
        _publish([curve])
