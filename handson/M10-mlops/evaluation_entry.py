"""
SageMaker Pipeline の評価ステップ用エントリースクリプト

学習済み XGBoost モデルをテストデータで評価し、AUC を含む評価レポートを
evaluation.json として出力します。Pipeline の ConditionStep はこの AUC を見て
モデルを Model Registry に登録するか判断します。

- モデル:   /opt/ml/processing/model/model.tar.gz
- テスト:   /opt/ml/processing/test/test.csv （先頭列がターゲット、ヘッダなし）
- 出力:     /opt/ml/processing/evaluation/evaluation.json
"""

import os
import json
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
EVAL_DIR = "/opt/ml/processing/evaluation"


def main():
    # モデルを展開
    with tarfile.open(os.path.join(MODEL_DIR, "model.tar.gz")) as tar:
        tar.extractall(path=MODEL_DIR)

    booster = xgb.Booster()
    # 組み込み XGBoost のモデルファイル名は xgboost または *.bin のことがある
    model_file = None
    for name in os.listdir(MODEL_DIR):
        if name in ("xgboost-model", "xgboost") or name.endswith((".bin", ".json", ".model")):
            model_file = os.path.join(MODEL_DIR, name)
            break
    if model_file is None:
        raise FileNotFoundError(f"モデルファイルが見つかりません: {os.listdir(MODEL_DIR)}")
    booster.load_model(model_file)

    df = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    prob = booster.predict(xgb.DMatrix(X))
    pred = (prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y, prob))
    acc = float(accuracy_score(y, pred))

    report = {
        "binary_classification_metrics": {
            "auc": {"value": auc},
            "accuracy": {"value": acc},
        }
    }

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f)

    print(f"[evaluation] AUC={auc:.4f}, Accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
