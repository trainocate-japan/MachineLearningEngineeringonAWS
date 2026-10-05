"""
モジュール 7: SageMaker 自動モデルチューニング（AMT）

Amazon SageMaker の HyperparameterTuner を使用して、
組み込み XGBoost の最適なハイパーパラメータをベイズ最適化で探索します。

扱う内容:
- ハイパーパラメータチューニング手法
  （グリッド検索 / ランダム検索 / ベイズ最適化 / Hyperband）
- SageMaker AI 自動モデルチューニング (AMT)

処理の流れ:
1. 特徴量データを train/validation に分割し S3 にアップロード
2. XGBoost Estimator と探索するハイパーパラメータ範囲を定義
3. HyperparameterTuner を validation:auc 最大化で実行
4. 最良ジョブのハイパーパラメータを取得

--dry-run で AWS を呼ばずに構成（探索手法の比較表含む）のみ表示します。
"""

import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
FEATURES_CSV = os.path.join(DATA_DIR, "features_income.csv")

PREFIX = "mle-tuning"
JOB_BASE = "mle-handson-amt"


def get_role() -> str:
    # 1) 環境変数が最優先
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role:
        return role
    # 2) IAM から SageMaker 実行ロールを名前で解決する。
    #    EC2 上で sagemaker.get_execution_role() を使うと EC2 インスタンスロールを
    #    返してしまい、sagemaker.amazonaws.com が assume できず Training が失敗する。
    try:
        import boto3

        iam = boto3.client("iam")
        return iam.get_role(RoleName="MLEngineeringHandsonSageMakerRole")["Role"]["Arn"]
    except Exception:
        pass
    # 3) 最後の手段
    import sagemaker

    return sagemaker.get_execution_role()


def print_strategy_table():
    print("\n  [ハイパーパラメータチューニング手法の比較]")
    rows = [
        ("グリッド検索", "すべての組み合わせを試す", "網羅的だが高コスト"),
        ("ランダム検索", "ランダムな組み合わせ", "効率的・並列向き"),
        ("ベイズ最適化", "回帰問題として過去結果を活用", "少ない試行で高精度（本デモ）"),
        ("Hyperband", "有望な構成にリソース集中", "反復アルゴリズム向き"),
    ]
    for name, how, note in rows:
        print(f"    - {name:12s}: {how}（{note}）")


def main():
    parser = argparse.ArgumentParser(description="SageMaker 自動モデルチューニング（AMT）")
    parser.add_argument("--dry-run", action="store_true", help="AWS を呼ばず構成のみ表示")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--max-jobs", type=int, default=8, help="総トレーニングジョブ数")
    parser.add_argument("--max-parallel", type=int, default=2, help="並列ジョブ数")
    args = parser.parse_args()

    print("=" * 60)
    print(" SageMaker 自動モデルチューニング（AMT）")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下の構成でチューニングを実行します:")
        print(f"  - 戦略: ベイズ最適化")
        print(f"  - 目的メトリクス: validation:auc（最大化）")
        print(f"  - 総ジョブ数: {args.max_jobs} / 並列: {args.max_parallel}")
        print(f"  - 探索範囲: max_depth[3-10], eta[0.01-0.3], "
              "min_child_weight[1-10], subsample[0.5-1.0]")
        print(f"  - ジョブ名プレフィックス: {JOB_BASE}")
        print_strategy_table()
        print("\n  実際に実行するには --dry-run を外してください。")
        return

    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {FEATURES_CSV}\n"
            "先に M04 の feature_transform.py を実行してください。"
        )

    import pandas as pd
    from sklearn.model_selection import train_test_split
    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput
    from sagemaker.tuner import (
        HyperparameterTuner,
        IntegerParameter,
        ContinuousParameter,
    )

    session = sagemaker.Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    role = get_role()

    print(f"\n  リージョン: {region}")
    print(f"  SageMaker ロール: {role}")
    print(f"  デフォルトバケット: {bucket}")

    # データ準備
    df = pd.read_csv(FEATURES_CSV)
    train, valid = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["income_high"]
    )
    local_dir = os.path.join(HERE, "output")
    os.makedirs(local_dir, exist_ok=True)
    train.to_csv(os.path.join(local_dir, "train.csv"), index=False, header=False)
    valid.to_csv(os.path.join(local_dir, "validation.csv"), index=False, header=False)
    train_s3 = session.upload_data(os.path.join(local_dir, "train.csv"),
                                   bucket=bucket, key_prefix=f"{PREFIX}/train")
    valid_s3 = session.upload_data(os.path.join(local_dir, "validation.csv"),
                                   bucket=bucket, key_prefix=f"{PREFIX}/validation")

    image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        output_path=f"s3://{bucket}/{PREFIX}/output",
        base_job_name=JOB_BASE,
        sagemaker_session=session,
    )
    estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        num_round=100,
    )

    # 探索するハイパーパラメータ範囲
    ranges = {
        "max_depth": IntegerParameter(3, 10),
        "eta": ContinuousParameter(0.01, 0.3),
        "min_child_weight": IntegerParameter(1, 10),
        "subsample": ContinuousParameter(0.5, 1.0),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:auc",
        objective_type="Maximize",
        hyperparameter_ranges=ranges,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel,
        strategy="Bayesian",
        base_tuning_job_name=JOB_BASE,
    )

    print("\n  チューニングジョブを実行中...（複数ジョブのため 10〜20 分）")
    tuner.fit(
        {
            "train": TrainingInput(train_s3, content_type="text/csv"),
            "validation": TrainingInput(valid_s3, content_type="text/csv"),
        }
    )

    best = tuner.best_estimator()
    print("\n  ✅ チューニング完了")
    print("  最良ジョブのハイパーパラメータ:")
    for k, v in best.hyperparameters().items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
