"""
モジュール 6: SageMaker トレーニングジョブ（組み込み XGBoost）

Amazon SageMaker の組み込み XGBoost アルゴリズムを使って、
マネージドなトレーニングジョブを実行します。

扱う内容:
- SageMaker AI でのモデルトレーニング
  （SageMaker Training / コンピュートオプション / 自動モデルチューニング）

処理の流れ:
1. 特徴量データを train/validation に分割し、ヘッダなし CSV で S3 にアップロード
2. 組み込み XGBoost Estimator を構成（ml.m5.large）
3. fit() でトレーニングジョブを実行
4. モデルアーティファクト（model.tar.gz）が S3 に保存される

生成されるジョブ名/モデルは cleanup_all.sh が削除できるよう "mle-handson" を含みます。
--dry-run で AWS を呼ばずに構成のみ表示します。
"""

import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
FEATURES_CSV = os.path.join(DATA_DIR, "features_income.csv")

PREFIX = "mle-training"
JOB_BASE = "mle-handson-xgb"


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


def prepare_and_upload(session, bucket):
    """特徴量 CSV を train/validation に分割し、ヘッダなしで S3 にアップロード"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(FEATURES_CSV)  # 先頭列が income_high（ターゲット）
    train, valid = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["income_high"]
    )

    local_dir = os.path.join(HERE, "output")
    os.makedirs(local_dir, exist_ok=True)
    train_path = os.path.join(local_dir, "train.csv")
    valid_path = os.path.join(local_dir, "validation.csv")
    # 組み込み XGBoost: ヘッダなし、先頭列ターゲット
    train.to_csv(train_path, index=False, header=False)
    valid.to_csv(valid_path, index=False, header=False)

    train_s3 = session.upload_data(train_path, bucket=bucket, key_prefix=f"{PREFIX}/train")
    valid_s3 = session.upload_data(valid_path, bucket=bucket, key_prefix=f"{PREFIX}/validation")
    return train_s3, valid_s3


def main():
    parser = argparse.ArgumentParser(description="SageMaker 組み込み XGBoost トレーニング")
    parser.add_argument("--dry-run", action="store_true", help="AWS を呼ばず構成のみ表示")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" SageMaker トレーニングジョブ（組み込み XGBoost）")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下の構成でトレーニングを実行します:")
        print(f"  - アルゴリズム: 組み込み XGBoost (binary:logistic)")
        print(f"  - インスタンスタイプ: {args.instance_type}")
        print(f"  - 入力: {FEATURES_CSV} を train/validation に分割")
        print(f"  - ジョブ名プレフィックス: {JOB_BASE}")
        print(f"  - S3 プレフィックス: {PREFIX}")
        print("  - ハイパーパラメータ: max_depth=4, eta=0.2, num_round=100")
        print("\n  実際に実行するには --dry-run を外してください。")
        return

    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {FEATURES_CSV}\n"
            "先に M04 の feature_transform.py を実行してください。"
        )

    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput

    session = sagemaker.Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    role = get_role()

    print(f"\n  リージョン: {region}")
    print(f"  SageMaker ロール: {role}")
    print(f"  デフォルトバケット: {bucket}")

    # データ準備・アップロード
    train_s3, valid_s3 = prepare_and_upload(session, bucket)
    print(f"  train:      {train_s3}")
    print(f"  validation: {valid_s3}")

    # 組み込み XGBoost コンテナイメージ URI を取得
    image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")
    output_path = f"s3://{bucket}/{PREFIX}/output"

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        output_path=output_path,
        base_job_name=JOB_BASE,
        sagemaker_session=session,
    )
    estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=4,
        eta=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        num_round=100,
    )

    print("\n  トレーニングジョブを実行中...（数分かかります）")
    estimator.fit(
        {
            "train": TrainingInput(train_s3, content_type="text/csv"),
            "validation": TrainingInput(valid_s3, content_type="text/csv"),
        }
    )

    print("\n  ✅ トレーニングジョブ完了")
    print(f"  モデルアーティファクト: {estimator.model_data}")
    print("\n  このモデルは M08（デプロイ）で使用できます。")
    print("  ジョブ名:", estimator.latest_training_job.name)


if __name__ == "__main__":
    main()
