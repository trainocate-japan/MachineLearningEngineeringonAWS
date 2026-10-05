"""
モジュール 4: SageMaker Processing ジョブ

ローカルで確認した変換処理を、Amazon SageMaker Processing の
マネージドジョブとしてスケール実行します。

扱う内容:
- AWS データ変換サービス
  - SageMaker Processing（フルマネージドなデータ処理・特徴量エンジニアリング）

処理の流れ:
1. 生データ（raw_income.csv）を S3 にアップロード
2. SKLearnProcessor で前処理スクリプト(processing_entry.py)を実行
3. 出力（train.csv / validation.csv / test.csv）を S3 に保存

実行には $SAGEMAKER_ROLE_ARN と SageMaker 実行環境が必要です。
--dry-run を付けると AWS を呼ばずに構成だけ表示します。
"""

import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_income.csv")
ENTRY = os.path.join(HERE, "processing_entry.py")

PREFIX = "mle-processing"


def get_role() -> str:
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role:
        return role
    import sagemaker

    return sagemaker.get_execution_role()


def main():
    parser = argparse.ArgumentParser(description="SageMaker Processing ジョブの起動")
    parser.add_argument("--dry-run", action="store_true", help="AWS を呼ばず構成のみ表示")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" SageMaker Processing: 特徴量エンジニアリングジョブ")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下の構成でジョブを実行します:")
        print(f"  - エントリースクリプト: {ENTRY}")
        print(f"  - 入力データ: {RAW_CSV}")
        print(f"  - インスタンスタイプ: {args.instance_type}")
        print(f"  - 出力: /opt/ml/processing/output/{{train,validation,test}}")
        print("  - 分割比率: train 70% / validation 15% / test 15%")
        print("\n  実際に実行するには --dry-run を外してください。")
        return

    import sagemaker
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.processing import ProcessingInput, ProcessingOutput

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {RAW_CSV}\n"
            "先に以下を実行してください: python ../common/generate_dataset.py"
        )

    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = get_role()
    print(f"\n  SageMaker ロール: {role}")
    print(f"  デフォルトバケット: {bucket}")

    # 1. 生データを S3 にアップロード
    input_s3 = session.upload_data(
        RAW_CSV, bucket=bucket, key_prefix=f"{PREFIX}/input"
    )
    print(f"  入力データをアップロード: {input_s3}")

    # 2. SKLearnProcessor を構成
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        base_job_name=PREFIX,
        sagemaker_session=session,
    )

    output_s3 = f"s3://{bucket}/{PREFIX}/output"

    # 3. ジョブを実行
    print("\n  Processing ジョブを実行中...（数分かかります）")
    processor.run(
        code=ENTRY,
        inputs=[
            ProcessingInput(source=input_s3, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output/train",
                             destination=f"{output_s3}/train"),
            ProcessingOutput(source="/opt/ml/processing/output/validation",
                             destination=f"{output_s3}/validation"),
            ProcessingOutput(source="/opt/ml/processing/output/test",
                             destination=f"{output_s3}/test"),
        ],
    )

    print("\n  ✅ Processing ジョブ完了")
    print(f"  出力先: {output_s3}/{{train,validation,test}}")
    print("\n  この出力は M06（モデルトレーニング）の入力として使用できます。")


if __name__ == "__main__":
    main()
