"""
モジュール 8: バッチ変換

大量データに対してオフライン推論を実行します。常時起動のエンドポイントが
不要なため、推論完了後は自動的にインスタンスが停止します（コスト効率が高い）。

扱う内容:
- モデル推論戦略の選択（バッチ変換）
- SageMaker AI モデルデプロイのベストプラクティス（バッチ変換）

処理の流れ:
1. モデルをトレーニング（_common）
2. Transformer を作成し、S3 のテストデータに対してバッチ推論
3. 出力（予測結果）を S3 から取得して確認

バッチ変換はジョブ完了後にリソースが解放されるため、
リアルタイムエンドポイントのような削除忘れによる課金が発生しません。

--dry-run で AWS を呼ばず構成のみ表示。
"""

import os
import argparse

import _common as C


def main():
    parser = argparse.ArgumentParser(description="バッチ変換ジョブ")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" バッチ変換（オフライン推論）")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下を実行します:")
        print("  1. 組み込み XGBoost でモデルをトレーニング")
        print("  2. Transformer でテストデータに対しバッチ推論")
        print("  3. 出力を S3 から取得")
        print("\n  リアルタイム vs バッチ:")
        print("    - リアルタイム: 常時起動・低レイテンシ・削除し忘れると課金継続")
        print("    - バッチ変換:   ジョブ完了で自動停止・大量データ向き・コスト効率")
        return

    C.require_features()

    import sagemaker

    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = C.get_role()
    print(f"\n  ロール: {role}\n  バケット: {bucket}")

    train_s3, valid_s3, test_s3 = C.prepare_data(session, bucket)
    print("  モデルをトレーニング中...")
    estimator = C.train_model(session, role, bucket, train_s3, valid_s3, args.instance_type)

    output_s3 = f"s3://{bucket}/{C.PREFIX}/batch-output"
    transformer = estimator.transformer(
        instance_count=1,
        instance_type=args.instance_type,
        output_path=output_s3,
        accept="text/csv",
        strategy="MultiRecord",
        assemble_with="Line",
    )

    print(f"\n  バッチ変換ジョブを実行中...（数分）\n  入力: {test_s3}")
    transformer.transform(
        test_s3,
        content_type="text/csv",
        split_type="Line",
    )
    transformer.wait()

    print("\n  ✅ バッチ変換完了")
    print(f"  出力先: {output_s3}")

    # 出力を取得して先頭を表示
    import boto3

    s3 = boto3.client("s3", region_name=session.boto_region_name)
    key_prefix = f"{C.PREFIX}/batch-output/"
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix).get("Contents", [])
    out_keys = [o["Key"] for o in objs if o["Key"].endswith(".out")]
    if out_keys:
        body = s3.get_object(Bucket=bucket, Key=out_keys[0])["Body"].read().decode()
        preds = body.strip().splitlines()[:5]
        print("\n  [推論結果（先頭 5 件・確率）]")
        for i, p in enumerate(preds, 1):
            prob = float(p)
            label = "高収入(1)" if prob >= 0.5 else "低収入(0)"
            print(f"    サンプル{i}: 確率={prob:.4f} → {label}")
    else:
        print("  （出力ファイルの取得に失敗。S3 を直接確認してください）")

    print("\n  ℹ️  バッチ変換はジョブ完了でインスタンスが解放されます（追加課金なし）。")


if __name__ == "__main__":
    main()
