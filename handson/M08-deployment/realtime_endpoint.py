"""
モジュール 8: リアルタイムエンドポイント

モデルを SageMaker リアルタイムエンドポイントにデプロイし、
低レイテンシ推論を実行します。

扱う内容:
- デプロイターゲットの選択（SageMaker エンドポイント）
- モデル推論戦略の選択（リアルタイム推論）

処理の流れ:
1. モデルをトレーニング（_common）
2. deploy() でリアルタイムエンドポイントを作成（mle-handson-rt-*）
3. テストデータで推論を実行
4. ★ 必ず最後にエンドポイントを削除（課金停止）

--dry-run で AWS を呼ばず構成のみ表示。
--keep を付けるとエンドポイントを削除せず残します（デモ継続用、課金注意）。
"""

import os
import argparse
import time

import _common as C


def main():
    parser = argparse.ArgumentParser(description="リアルタイムエンドポイントのデプロイ")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true", help="エンドポイントを削除せず残す（課金注意）")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" リアルタイムエンドポイントのデプロイ")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下を実行します:")
        print("  1. 組み込み XGBoost でモデルをトレーニング")
        print(f"  2. ml.m5.large のリアルタイムエンドポイントにデプロイ")
        print("  3. テストデータで推論")
        print("  4. エンドポイント削除（--keep 指定時は残す）")
        print("\n  推論オプションの比較:")
        print("    - リアルタイム: 低レイテンシ・常時起動（本デモ）")
        print("    - サーバーレス: 断続的トラフィック・自動スケール")
        print("    - 非同期: 大きなペイロード・長い処理時間")
        print("    - バッチ変換: 大量データのオフライン推論")
        return

    C.require_features()

    import sagemaker
    from sagemaker.serializers import CSVSerializer

    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = C.get_role()
    print(f"\n  ロール: {role}\n  バケット: {bucket}")

    train_s3, valid_s3, _ = C.prepare_data(session, bucket)
    print("  モデルをトレーニング中...")
    estimator = C.train_model(session, role, bucket, train_s3, valid_s3, args.instance_type)

    endpoint_name = f"{C.PREFIX}-rt-{int(time.time())}"
    print(f"\n  エンドポイントをデプロイ中: {endpoint_name}（数分）")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=endpoint_name,
        serializer=CSVSerializer(),
    )

    try:
        # テストデータの先頭数件で推論
        test_path = os.path.join(C.HERE, "output", "test_features.csv")
        with open(test_path) as f:
            rows = [line.strip() for line in f.readlines()[:5]]

        print("\n  [推論結果（先頭 5 件）]")
        for i, row in enumerate(rows, 1):
            result = predictor.predict(row).decode("utf-8").strip()
            prob = float(result)
            label = "高収入(1)" if prob >= 0.5 else "低収入(0)"
            print(f"    サンプル{i}: 確率={prob:.4f} → 予測={label}")

        print("\n  ✅ リアルタイム推論成功")
    finally:
        if args.keep:
            print(f"\n  ⚠️  --keep 指定のためエンドポイントを残します: {endpoint_name}")
            print("     課金が続きます。デモ後は必ず削除してください:")
            print(f"     aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")
        else:
            print(f"\n  エンドポイントを削除中: {endpoint_name}")
            predictor.delete_endpoint()
            print("  ✅ エンドポイント削除完了（課金停止）")


if __name__ == "__main__":
    main()
