"""
モジュール 10: SageMaker Pipelines

データ処理 → トレーニング → 評価 → 条件付きモデル登録を 1 本のパイプラインに
まとめ、ML ワークフローを自動化・再現可能にします。

扱う内容:
- MLOps の基本（CI/CD・再現性・自動化）
- SageMaker Pipelines / SageMaker Projects
- Model Registry

パイプライン構成:
  [ProcessingStep]  raw_income.csv を前処理し train/validation/test に分割
        │
  [TrainingStep]    組み込み XGBoost で学習
        │
  [ProcessingStep]  テストデータで評価（evaluation.json に AUC）
        │
  [ConditionStep]   AUC >= 閾値 なら
        │
  [RegisterModel]   Model Registry（mle-handson-model-group）に登録

--dry-run で AWS を呼ばず構成のみ表示。
--register-only で既存パイプラインを再作成せず定義の upsert のみ。
"""

import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_income.csv")
PROCESSING_ENTRY = os.path.join(
    HERE, "..", "M04-feature-engineering", "processing_entry.py"
)
EVAL_ENTRY = os.path.join(HERE, "evaluation_entry.py")

PIPELINE_NAME = "mle-handson-pipeline"
MODEL_GROUP = "mle-handson-model-group"
PREFIX = "mle-handson"
AUC_THRESHOLD = 0.80


def get_role() -> str:
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role:
        return role
    import sagemaker

    return sagemaker.get_execution_role()


def build_pipeline(session, role, bucket):
    import sagemaker
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep
    from sagemaker.workflow.step_collection import RegisterModel
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.functions import JsonGet
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.parameters import ParameterString, ParameterFloat
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput
    from sagemaker.model_metrics import ModelMetrics, MetricsSource

    region = session.boto_region_name
    xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

    # パイプラインパラメータ
    input_data = ParameterString(
        name="InputData",
        default_value=session.upload_data(
            RAW_CSV, bucket=bucket, key_prefix=f"{PREFIX}/pipeline/input"
        ),
    )
    auc_threshold = ParameterFloat(name="AucThreshold", default_value=AUC_THRESHOLD)

    # --- Step 1: 前処理 ---
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{PREFIX}-pipe-process",
        sagemaker_session=session,
    )
    step_process = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
        ],
        code=PROCESSING_ENTRY,
    )

    # --- Step 2: トレーニング ---
    estimator = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://{bucket}/{PREFIX}/pipeline/model",
        base_job_name=f"{PREFIX}-pipe-train",
        sagemaker_session=session,
    )
    estimator.set_hyperparameters(
        objective="binary:logistic", eval_metric="auc",
        max_depth=4, eta=0.2, num_round=100,
    )
    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # --- Step 3: 評価 ---
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{PREFIX}-pipe-eval",
        sagemaker_session=session,
    )
    eval_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=EVAL_ENTRY,
        property_files=[eval_report],
    )

    # --- Step 4: 条件付き登録 ---
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    cond = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=eval_report,
            json_path="binary_classification_metrics.auc.value",
        ),
        right=auc_threshold,
    )
    step_cond = ConditionStep(
        name="CheckAUC",
        conditions=[cond],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data, auc_threshold],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser(description="SageMaker Pipeline の作成・実行")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--register-only", action="store_true",
                        help="定義の upsert のみ（実行しない）")
    args = parser.parse_args()

    print("=" * 60)
    print(" SageMaker Pipelines: ML ワークフローの自動化")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下のパイプラインを作成します:")
        print(f"  パイプライン名: {PIPELINE_NAME}")
        print(f"  Model Package Group: {MODEL_GROUP}")
        print(f"  AUC 閾値: {AUC_THRESHOLD}")
        print("\n  ステップ:")
        print("    1. PreprocessData   : 前処理（train/validation/test 分割）")
        print("    2. TrainModel       : 組み込み XGBoost で学習")
        print("    3. EvaluateModel    : テストデータで AUC を評価")
        print("    4. CheckAUC         : AUC >= 閾値 なら →")
        print("    5. RegisterModel    : Model Registry に登録(承認待ち)")
        print("\n  実際に作成・実行するには --dry-run を外してください。")
        return

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {RAW_CSV}\n"
            "先に python ../common/generate_dataset.py を実行してください。"
        )

    import sagemaker

    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = get_role()
    print(f"\n  ロール: {role}\n  バケット: {bucket}")

    pipeline = build_pipeline(session, role, bucket)
    pipeline.upsert(role_arn=role)
    print(f"\n  ✅ パイプラインを upsert: {PIPELINE_NAME}")

    if args.register_only:
        print("  --register-only 指定のため実行はスキップします。")
        print("  実行する場合: SageMaker Studio か pipeline.start() を使用してください。")
        return

    print("\n  パイプラインを実行中...（10〜20 分）")
    execution = pipeline.start()
    execution.wait()
    print("\n  ✅ パイプライン実行完了")
    for step in execution.list_steps():
        print(f"    - {step['StepName']}: {step['StepStatus']}")


if __name__ == "__main__":
    main()
