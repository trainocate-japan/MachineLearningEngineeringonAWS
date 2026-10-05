"""
M08 共通ヘルパー

デプロイ用のモデルを用意するための補助関数群。
特徴量データを S3 にアップロードし、組み込み XGBoost でトレーニングした
Estimator を返します（デプロイ系スクリプトから再利用）。
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
FEATURES_CSV = os.path.join(DATA_DIR, "features_income.csv")

PREFIX = "mle-handson"


def get_role() -> str:
    # 1) 環境変数が最優先
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role:
        return role
    # 2) IAM から SageMaker 実行ロールを名前で解決する。
    #    EC2 上で sagemaker.get_execution_role() を使うと EC2 インスタンスロールを
    #    返してしまい、sagemaker.amazonaws.com が assume できず失敗する。
    try:
        import boto3

        iam = boto3.client("iam")
        return iam.get_role(RoleName="MLEngineeringHandsonSageMakerRole")["Role"]["Arn"]
    except Exception:
        pass
    # 3) 最後の手段
    import sagemaker

    return sagemaker.get_execution_role()


def require_features():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {FEATURES_CSV}\n"
            "先に以下を実行してください:\n"
            "  python ../common/generate_dataset.py\n"
            "  python ../M04-feature-engineering/data_cleaning.py\n"
            "  python ../M04-feature-engineering/feature_transform.py"
        )


def prepare_data(session, bucket):
    """特徴量を train/validation/test に分割し S3 にアップロード。
    test は推論（バッチ変換・エンドポイント呼び出し）用に特徴量のみ（ターゲット除外）も用意。
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(FEATURES_CSV)
    train, temp = train_test_split(df, test_size=0.3, random_state=42,
                                   stratify=df["income_high"])
    valid, test = train_test_split(temp, test_size=0.5, random_state=42,
                                    stratify=temp["income_high"])

    local = os.path.join(HERE, "output")
    os.makedirs(local, exist_ok=True)

    train.to_csv(os.path.join(local, "train.csv"), index=False, header=False)
    valid.to_csv(os.path.join(local, "validation.csv"), index=False, header=False)
    # 推論入力: ターゲット列を除いた特徴量のみ（ヘッダなし）
    test.drop(columns=["income_high"]).to_csv(
        os.path.join(local, "test_features.csv"), index=False, header=False
    )
    # 正解ラベル（評価用）
    test[["income_high"]].to_csv(os.path.join(local, "test_labels.csv"), index=False)

    train_s3 = session.upload_data(os.path.join(local, "train.csv"),
                                   bucket=bucket, key_prefix=f"{PREFIX}/train")
    valid_s3 = session.upload_data(os.path.join(local, "validation.csv"),
                                   bucket=bucket, key_prefix=f"{PREFIX}/validation")
    test_s3 = session.upload_data(os.path.join(local, "test_features.csv"),
                                  bucket=bucket, key_prefix=f"{PREFIX}/test")
    return train_s3, valid_s3, test_s3


def train_model(session, role, bucket, train_s3, valid_s3,
                instance_type="ml.m5.large"):
    """組み込み XGBoost でトレーニングし、学習済み Estimator を返す。"""
    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput

    region = session.boto_region_name
    image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        output_path=f"s3://{bucket}/{PREFIX}/output",
        base_job_name=f"{PREFIX}-deploy-train",
        sagemaker_session=session,
    )
    estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=4,
        eta=0.2,
        num_round=100,
    )
    estimator.fit(
        {
            "train": TrainingInput(train_s3, content_type="text/csv"),
            "validation": TrainingInput(valid_s3, content_type="text/csv"),
        }
    )
    return estimator
