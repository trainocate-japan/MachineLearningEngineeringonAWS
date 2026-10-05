"""
モジュール 11: SageMaker Model Monitor（ベースライン作成 + モニタリングスケジュール）

トレーニングデータからベースライン（統計量と制約）を作成し、
データキャプチャ付きエンドポイントに対してデータ品質モニタリングスケジュールを
設定します。

扱う内容:
- SageMaker Model Monitor
- データ品質ドリフトのモニタリング手順
  （ステップ 2: ベースライン作成 / ステップ 3: ジョブの定義とスケジュール /
    ステップ 4: CloudWatch 統合）

前提: data_capture.py を --keep 付きで実行し、エンドポイントが稼働していること。

処理の流れ:
1. ベースラインジョブを実行（statistics.json / constraints.json を生成）
2. 時間単位のデータ品質モニタリングスケジュールを作成
3. CloudWatch 統合とスケジュール確認

--dry-run で AWS を呼ばず構成のみ表示。
"""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "M08-deployment"))

PREFIX = "mle-handson"
SCHEDULE_NAME = "mle-handson-monitor-schedule"


def main():
    parser = argparse.ArgumentParser(description="SageMaker Model Monitor の設定")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" SageMaker Model Monitor")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下を実行します:")
        print("  1. ベースラインジョブ: トレーニングデータから統計量と制約を生成")
        print("     → statistics.json（記述統計） / constraints.json（制約）")
        print("  2. データ品質モニタリングスケジュールを作成（時間単位）")
        print(f"     スケジュール名: {SCHEDULE_NAME}")
        print("  3. CloudWatch にメトリクスを発行、違反時にアラーム")
        print("\n  違反チェックの例（constraint_violations.json）:")
        print("    data_type_check / completeness_check / baseline_drift_check /")
        print("    missing_column_check / extra_column_check / categorical_values_check")
        return

    endpoint_file = os.path.join(HERE, "endpoint_name.txt")
    if not os.path.exists(endpoint_file):
        print("\n  ⚠️  endpoint_name.txt が見つかりません。")
        print("  先に以下を実行してエンドポイントを稼働させてください:")
        print("    python data_capture.py --keep")
        return

    with open(endpoint_file) as f:
        endpoint_name = f.read().strip()

    import _common as C
    import sagemaker
    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    from sagemaker.model_monitor import CronExpressionGenerator, EndpointInput

    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = C.get_role()
    print(f"\n  ロール: {role}\n  バケット: {bucket}\n  エンドポイント: {endpoint_name}")

    # ベースライン入力: トレーニングデータ（ヘッダ付きで用意）
    C.require_features()
    import pandas as pd

    df = pd.read_csv(C.FEATURES_CSV)
    baseline_dir = os.path.join(HERE, "output")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_csv = os.path.join(baseline_dir, "baseline.csv")
    df.to_csv(baseline_csv, index=False, header=True)  # ベースラインはヘッダ付き
    baseline_s3 = session.upload_data(baseline_csv, bucket=bucket,
                                      key_prefix=f"{PREFIX}/baseline-input")

    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=session,
    )

    baseline_results = f"s3://{bucket}/{PREFIX}/baseline-results"
    print("\n  ベースラインジョブを実行中...（数分）")
    monitor.suggest_baseline(
        baseline_dataset=baseline_s3,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results,
        wait=True,
    )
    print(f"  ✅ ベースライン作成完了: {baseline_results}")
    print("     statistics.json / constraints.json を確認してください。")

    # モニタリングスケジュール作成（時間単位）
    reports_s3 = f"s3://{bucket}/{PREFIX}/monitor-reports"
    print(f"\n  モニタリングスケジュールを作成中: {SCHEDULE_NAME}")
    monitor.create_monitoring_schedule(
        monitor_schedule_name=SCHEDULE_NAME,
        endpoint_input=EndpointInput(
            endpoint_name=endpoint_name,
            destination="/opt/ml/processing/input/endpoint",
        ),
        output_s3_uri=reports_s3,
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )

    print("\n  ✅ モニタリングスケジュール作成完了")
    print(f"  レポート出力先: {reports_s3}")
    print("\n  [次のステップ]")
    print("  - 最初のモニタリングジョブは次の正時に自動実行されます")
    print("  - CloudWatch メトリクスで違反を監視、アラーム → SNS 通知を設定")
    print("  - 完了後は cleanup_all.sh でスケジュールとエンドポイントを削除")


if __name__ == "__main__":
    main()
