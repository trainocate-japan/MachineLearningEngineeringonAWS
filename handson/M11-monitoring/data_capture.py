"""
モジュール 11: データキャプチャ付きエンドポイント

リアルタイムエンドポイントにデータキャプチャを設定し、推論リクエストと
レスポンスを S3 に記録します。これは Model Monitor がドリフトを検出するための
入力になります。

扱う内容:
- データ品質ドリフトのモニタリング手順
  （ステップ 1: データキャプチャをエンドポイントで開始）

処理の流れ:
1. モデルをトレーニング（M08 の _common を再利用）
2. DataCaptureConfig 付きでエンドポイントをデプロイ（キャプチャ率 100%）
3. テストデータで推論し、S3 にリクエスト/レスポンスを記録
4. ★ 課金のため、デモ後はエンドポイントを削除（--keep で保持可）

--dry-run で AWS を呼ばず構成のみ表示。

※ M08 の _common.py を利用するため、M08 ディレクトリをパスに追加します。
"""

import os
import sys
import argparse
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "M08-deployment"))

PREFIX = "mle-handson"


def main():
    parser = argparse.ArgumentParser(description="データキャプチャ付きエンドポイント")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true",
                        help="エンドポイントを残す（Model Monitor で使うため。課金注意）")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" データキャプチャ付きエンドポイント")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下を実行します:")
        print("  1. 組み込み XGBoost でモデルをトレーニング")
        print("  2. DataCaptureConfig(sampling=100%) 付きでデプロイ")
        print("  3. テストデータで推論 → S3 にキャプチャ")
        print("\n  キャプチャ内容: 推論リクエスト(入力)とレスポンス(予測)")
        print("  → Model Monitor がこのキャプチャをベースラインと比較します")
        return

    import _common as C
    import sagemaker
    from sagemaker.model_monitor import DataCaptureConfig
    from sagemaker.serializers import CSVSerializer

    C.require_features()
    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = C.get_role()
    print(f"\n  ロール: {role}\n  バケット: {bucket}")

    train_s3, valid_s3, _ = C.prepare_data(session, bucket)
    print("  モデルをトレーニング中...")
    estimator = C.train_model(session, role, bucket, train_s3, valid_s3, args.instance_type)

    capture_s3 = f"s3://{bucket}/{PREFIX}/datacapture"
    endpoint_name = f"{PREFIX}-monitor-{int(time.time())}"

    data_capture = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=capture_s3,
    )

    print(f"\n  データキャプチャ付きでデプロイ中: {endpoint_name}")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=endpoint_name,
        data_capture_config=data_capture,
        serializer=CSVSerializer(),
    )

    try:
        test_path = os.path.join(C.HERE, "output", "test_features.csv")
        with open(test_path) as f:
            rows = [line.strip() for line in f.readlines()[:50]]
        print(f"\n  {len(rows)} 件のリクエストを送信してキャプチャを生成中...")
        for row in rows:
            predictor.predict(row)
        print(f"  ✅ キャプチャ先: {capture_s3}")
        print("     （S3 に反映されるまで数分かかる場合があります）")

        # Model Monitor で使うためエンドポイント名を保存
        with open(os.path.join(HERE, "endpoint_name.txt"), "w") as f:
            f.write(endpoint_name)
        print(f"\n  エンドポイント名を保存: endpoint_name.txt（model_monitor.py が参照）")
    finally:
        if args.keep:
            print(f"\n  ⚠️  --keep 指定のためエンドポイントを残します: {endpoint_name}")
            print("     model_monitor.py 実行後は必ず cleanup_all.sh で削除してください。")
        else:
            print(f"\n  エンドポイントを削除中: {endpoint_name}")
            predictor.delete_endpoint()
            print("  ✅ エンドポイント削除完了（課金停止）")
            print("  ※ Model Monitor デモを続ける場合は --keep を付けて再実行してください。")


if __name__ == "__main__":
    main()
