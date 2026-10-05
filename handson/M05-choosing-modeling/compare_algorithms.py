"""
モジュール 5: モデリングアプローチの比較

「どのアルゴリズムを選ぶか」を、実際に SageMaker トレーニングジョブを走らせて
比較します。同じ収入予測データに対して以下の 2 つの組み込みアルゴリズムを学習し、
検証 AUC・学習時間を比較します。

- Linear Learner : 線形モデル。高速・解釈しやすい・ベースライン向き
- XGBoost        : 勾配ブースティング。非線形で高精度になりやすい

扱う内容:
- モデリングアプローチの選択（組み込みアルゴリズムの比較）
- 解釈可能性 vs 精度、学習時間・コストのトレードオフ

処理の流れ:
1. 特徴量データを train/validation に分割し S3 にアップロード
2. Linear Learner と XGBoost をそれぞれトレーニング
3. 検証 AUC と学習時間を取得して比較表を表示

--dry-run で AWS を呼ばず構成のみ表示。
--algo で片方だけ実行も可能（linear / xgboost / both、既定 both）。
"""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
FEATURES_CSV = os.path.join(DATA_DIR, "features_income.csv")

PREFIX = "mle-handson"
FEATURE_DIM_PLACEHOLDER = None  # 実行時にデータから算出


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


def prepare_data(session, bucket):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(FEATURES_CSV)
    feature_dim = df.shape[1] - 1  # income_high を除いた特徴量数
    train, valid = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["income_high"]
    )
    local = os.path.join(HERE, "output")
    os.makedirs(local, exist_ok=True)
    train.to_csv(os.path.join(local, "train.csv"), index=False, header=False)
    valid.to_csv(os.path.join(local, "validation.csv"), index=False, header=False)
    train_s3 = session.upload_data(os.path.join(local, "train.csv"),
                                   bucket=bucket, key_prefix=f"{PREFIX}/compare/train")
    valid_s3 = session.upload_data(os.path.join(local, "validation.csv"),
                                   bucket=bucket, key_prefix=f"{PREFIX}/compare/validation")
    return train_s3, valid_s3, feature_dim


def train_xgboost(session, role, bucket, train_s3, valid_s3, instance_type):
    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput

    region = session.boto_region_name
    image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")
    est = Estimator(
        image_uri=image, role=role, instance_count=1, instance_type=instance_type,
        output_path=f"s3://{bucket}/{PREFIX}/compare/xgb",
        base_job_name=f"{PREFIX}-cmp-xgb", sagemaker_session=session,
    )
    est.set_hyperparameters(objective="binary:logistic", eval_metric="auc",
                            max_depth=4, eta=0.2, num_round=100)
    est.fit({"train": TrainingInput(train_s3, content_type="text/csv"),
             "validation": TrainingInput(valid_s3, content_type="text/csv")}, logs=False)
    return est


def train_linear(session, role, bucket, train_s3, valid_s3, instance_type, feature_dim):
    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput

    region = session.boto_region_name
    image = sagemaker.image_uris.retrieve("linear-learner", region)
    est = Estimator(
        image_uri=image, role=role, instance_count=1, instance_type=instance_type,
        output_path=f"s3://{bucket}/{PREFIX}/compare/linear",
        base_job_name=f"{PREFIX}-cmp-linear", sagemaker_session=session,
    )
    est.set_hyperparameters(
        predictor_type="binary_classifier",
        feature_dim=feature_dim,
        binary_classifier_model_selection_criteria="accuracy",
        mini_batch_size=200,
    )
    est.fit({"train": TrainingInput(train_s3, content_type="text/csv"),
             "validation": TrainingInput(valid_s3, content_type="text/csv")}, logs=False)
    return est


def _final_metric(session, job_name, metric_substring):
    """トレーニングジョブの最終メトリクス値を取得（なければ None）。"""
    desc = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
    for m in desc.get("FinalMetricDataList", []):
        if metric_substring in m["MetricName"]:
            return m["Value"]
    return None


def _duration(session, job_name):
    desc = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
    start = desc.get("TrainingStartTime")
    end = desc.get("TrainingEndTime")
    if start and end:
        return (end - start).total_seconds()
    return None


def main():
    parser = argparse.ArgumentParser(description="モデリングアプローチの比較（Linear vs XGBoost）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--algo", choices=["linear", "xgboost", "both"], default="both")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" モデリングアプローチの比較: Linear Learner vs XGBoost")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下を実行します:")
        print("  1. 特徴量データを train/validation に分割し S3 へ")
        print(f"  2. 対象アルゴリズム: {args.algo}")
        print("     - Linear Learner（線形・高速・解釈しやすい）")
        print("     - XGBoost（非線形・高精度）")
        print("  3. 検証 AUC と学習時間を比較")
        print("\n  観点: 解釈可能性 vs 精度 / 学習時間・コスト")
        print("  → 構造化データの二項分類では、まず両者を回して比較するのが実践的です。")
        return

    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {FEATURES_CSV}\n"
            "先に以下を実行してください:\n"
            "  python ../common/generate_dataset.py\n"
            "  python ../M04-feature-engineering/data_cleaning.py\n"
            "  python ../M04-feature-engineering/feature_transform.py"
        )

    import sagemaker

    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = get_role()
    print(f"\n  ロール: {role}\n  バケット: {bucket}")

    train_s3, valid_s3, feature_dim = prepare_data(session, bucket)
    print(f"  特徴量次元: {feature_dim}")

    results = []

    if args.algo in ("xgboost", "both"):
        print("\n  XGBoost をトレーニング中...（数分）")
        est = train_xgboost(session, role, bucket, train_s3, valid_s3, args.instance_type)
        job = est.latest_training_job.name
        results.append(("XGBoost", _final_metric(session, job, "auc"), _duration(session, job)))

    if args.algo in ("linear", "both"):
        print("\n  Linear Learner をトレーニング中...（数分）")
        est = train_linear(session, role, bucket, train_s3, valid_s3, args.instance_type, feature_dim)
        job = est.latest_training_job.name
        # Linear Learner の検証メトリクスは binary_classification_accuracy など
        auc = _final_metric(session, job, "auc") or _final_metric(session, job, "accuracy")
        results.append(("Linear Learner", auc, _duration(session, job)))

    print("\n" + "=" * 60)
    print(" 比較結果")
    print("=" * 60)
    print(f"  {'アルゴリズム':<18}{'検証スコア':<14}{'学習時間(秒)':<12}")
    print("  " + "-" * 44)
    for name, score, dur in results:
        s = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        d = f"{dur:.0f}" if isinstance(dur, (int, float)) else "N/A"
        print(f"  {name:<18}{s:<14}{d:<12}")

    print("\n  [考察]")
    print("  - Linear Learner: 高速・解釈しやすい。ベースラインや規制対応に向く")
    print("  - XGBoost: 非線形パターンを捉え高精度になりやすいが解釈性は下がる")
    print("  - スコアが拮抗するなら、よりシンプル/高速な方を選ぶのが定石")
    print("\n  ✅ 比較完了")


if __name__ == "__main__":
    main()
