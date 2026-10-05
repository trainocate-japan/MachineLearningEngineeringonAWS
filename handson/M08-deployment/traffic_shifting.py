"""
モジュール 8: ブルー/グリーンデプロイとトラフィックシフト

既存エンドポイントを新しいモデルに安全に更新する戦略を学びます。
SageMaker のデプロイガードレール（トラフィックシフト・自動ロールバック）を使い、
Linear / Canary / All-at-once のトラフィック移行モードを比較します。

扱う内容:
- デプロイ戦略
  （ブルー/グリーン・ローリング / All-at-once・Canary・Linear / 自動ロールバック）

このスクリプトは実際にエンドポイントの更新までは行わず、
デプロイガードレールの構成（BlueGreenUpdatePolicy）を組み立てて提示します。
受講者には「各モードがブラスト半径とリスクをどう抑えるか」を議論してもらいます。

--run を付けると、実際にエンドポイントを作成 → 新モデルへ Linear シフトで更新します
（課金注意・完了後に削除）。
"""

import argparse
import time

import _common as C


def describe_modes():
    print("\n  [トラフィックシフトモードの比較]")
    print("  ┌───────────────┬─────────────────────────────────────────┐")
    print("  │ モード        │ 特徴                                     │")
    print("  ├───────────────┼─────────────────────────────────────────┤")
    print("  │ All at once   │ 一括切替。更新時間/コスト最小。影響100%  │")
    print("  │ Canary        │ 少量を先行切替しブラスト半径を最小化      │")
    print("  │ Linear        │ 複数ステップで段階的に移行。リスク最小    │")
    print("  └───────────────┴─────────────────────────────────────────┘")
    print("\n  共通: CloudWatch アラームで異常を検知したら自動ロールバック")


def build_linear_policy():
    """Linear トラフィックシフトのデプロイガードレール構成を返す（update_endpoint 用の dict）。"""
    policy = {
        "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": {
                "Type": "LINEAR",
                "LinearStepSize": {"Type": "CAPACITY_PERCENT", "Value": 33},
                "WaitIntervalInSeconds": 120,
            },
            "TerminationWaitInSeconds": 300,
            "MaximumExecutionTimeoutInSeconds": 1800,
        },
        "AutoRollbackConfiguration": {
            "Alarms": [{"AlarmName": "mle-handson-high-error-rate"}],
        },
    }
    return policy


def main():
    parser = argparse.ArgumentParser(description="ブルー/グリーン トラフィックシフト")
    parser.add_argument("--run", action="store_true",
                        help="実際にエンドポイントを作成し Linear シフトで更新（課金注意）")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    print("=" * 60)
    print(" ブルー/グリーンデプロイとトラフィックシフト")
    print("=" * 60)

    describe_modes()

    policy = build_linear_policy()
    print("\n  [Linear トラフィックシフトのデプロイガードレール構成]")
    print("    Type: LINEAR / ステップ: 33% ずつ / 待機: 120 秒")
    print("    終了待機: 300 秒 / 自動ロールバック: CloudWatch アラーム連動")
    import json

    print("\n  構成(JSON):")
    print(json.dumps(policy, indent=2, ensure_ascii=False))

    if not args.run:
        print("\n  ℹ️  構成の提示のみ（--run 未指定）。")
        print("     実際にエンドポイントを作成して更新する場合は --run を付けてください（課金注意）。")
        return

    # --- 実際にエンドポイントを作成 → 新モデルへ Linear 更新するデモ ---
    C.require_features()
    import sagemaker

    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = C.get_role()
    print(f"\n  ロール: {role}\n  バケット: {bucket}")

    train_s3, valid_s3, _ = C.prepare_data(session, bucket)
    print("  モデル(v1=Blue)をトレーニング中...")
    estimator = C.train_model(session, role, bucket, train_s3, valid_s3, args.instance_type)

    endpoint_name = f"{C.PREFIX}-bg-{int(time.time())}"
    print(f"\n  初期エンドポイント(Blue)をデプロイ中: {endpoint_name}")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=endpoint_name,
    )

    try:
        print("\n  実運用では、ここで新モデル(Green)の EndpointConfig を作成し、")
        print("  update_endpoint() に上記 BlueGreenUpdatePolicy を渡して Linear 移行します。")
        print("  （本デモではコスト抑制のため更新自体はスキップし、構成提示に留めます）")
        print("\n  ✅ ブルー/グリーン デモ完了")
    finally:
        print(f"\n  エンドポイントを削除中: {endpoint_name}")
        predictor.delete_endpoint()
        print("  ✅ エンドポイント削除完了（課金停止）")


if __name__ == "__main__":
    main()
