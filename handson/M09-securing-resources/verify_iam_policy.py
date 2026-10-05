"""
モジュール 9: IAM ポリシーの実検証

最小権限ポリシーを「表示するだけ」でなく、IAM Policy Simulator
(simulate-custom-policy) で実際に許可/拒否を評価して確認します。
さらに、デプロイ済みの SageMaker 実行ロールに付与された権限も検証します。

扱う内容:
- アクセスコントロール（IAM ポリシー / ロール）
- 最小権限の検証（許可すべき操作は Allow、許可すべきでない操作は Deny）

処理の流れ:
1. ML 向け最小権限ポリシー（テスト用）を定義
2. simulate-custom-policy で代表的なアクションの許可/拒否を評価
3. （任意）デプロイ済み SageMaker ロールに対して simulate-principal-policy を実行

--dry-run で AWS を呼ばず、検証対象のアクション一覧のみ表示。
--role-arn を指定すると、そのロールの実効権限も検証します。
"""

import os
import argparse
import json

PREFIX = "mle-handson"

# 検証に使う最小権限ポリシー（テスト用）
TEST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": ["arn:aws:s3:::ml-data-bucket/*"],
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpoint",
            ],
            "Resource": "*",
        },
    ],
}

# (アクション, リソース, 期待される結果) のテストケース
TEST_CASES = [
    ("s3:GetObject", "arn:aws:s3:::ml-data-bucket/train.csv", "allowed"),
    ("s3:PutObject", "arn:aws:s3:::ml-data-bucket/out.csv", "allowed"),
    ("s3:DeleteBucket", "arn:aws:s3:::ml-data-bucket", "denied"),
    ("sagemaker:CreateTrainingJob", "*", "allowed"),
    ("iam:CreateUser", "*", "denied"),
    ("ec2:TerminateInstances", "*", "denied"),
]


def simulate_custom(iam):
    """テスト用ポリシーに対して各アクションを評価する。"""
    results = []
    for action, resource, expected in TEST_CASES:
        resp = iam.simulate_custom_policy(
            PolicyInputList=[json.dumps(TEST_POLICY)],
            ActionNames=[action],
            ResourceArns=[resource] if resource != "*" else ["*"],
        )
        decision = resp["EvaluationResults"][0]["EvalDecision"]
        # allowed / implicitDeny / explicitDeny
        got = "allowed" if decision == "allowed" else "denied"
        ok = got == expected
        results.append((action, resource, expected, decision, ok))
    return results


def main():
    parser = argparse.ArgumentParser(description="IAM ポリシーの実検証")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ROLE_ARN"),
                        help="検証する IAM ロール ARN（既定: $SAGEMAKER_ROLE_ARN）")
    args = parser.parse_args()

    print("=" * 60)
    print(" IAM ポリシーの実検証（Policy Simulator）")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] 以下のアクションを simulate-custom-policy で評価します:")
        for action, resource, expected in TEST_CASES:
            print(f"    - {action:32s} 期待: {expected}")
        print("\n  --role-arn 指定時は、そのロールの実効権限も検証します。")
        return

    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    iam = boto3.client("iam", region_name=region)

    # 1. カスタムポリシーのシミュレーション
    print("\n  [1] 最小権限ポリシーの許可/拒否を評価")
    print(f"  {'アクション':<34}{'期待':<8}{'結果':<14}{'判定'}")
    print("  " + "-" * 62)
    all_ok = True
    for action, resource, expected, decision, ok in simulate_custom(iam):
        mark = "✅" if ok else "⚠️ 不一致"
        if not ok:
            all_ok = False
        print(f"  {action:<34}{expected:<8}{decision:<14}{mark}")

    print("\n  → 許可すべき操作は allowed、許可すべきでない操作は denied になっていることを確認")
    if all_ok:
        print("  ✅ すべてのテストケースが期待どおり")
    else:
        print("  ⚠️  期待と異なる結果があります。ポリシーを見直してください。")

    # 2. 実ロールの検証（任意）
    if args.role_arn:
        print(f"\n  [2] デプロイ済みロールの実効権限を検証: {args.role_arn}")
        try:
            checks = [
                ("sagemaker:CreateTrainingJob", "*", "allowed"),
                ("s3:GetObject", "*", "allowed"),
                ("iam:DeleteRole", "*", "denied"),
            ]
            print(f"  {'アクション':<34}{'期待':<8}{'結果'}")
            print("  " + "-" * 54)
            for action, resource, expected in checks:
                resp = iam.simulate_principal_policy(
                    PolicySourceArn=args.role_arn,
                    ActionNames=[action],
                    ResourceArns=["*"],
                )
                decision = resp["EvaluationResults"][0]["EvalDecision"]
                print(f"  {action:<34}{expected:<8}{decision}")
            print("\n  ℹ️  SageMaker 実行ロールは SageMaker/S3 を許可し、"
                  "破壊的な IAM 操作は許可しない想定です。")
        except Exception as e:
            print(f"  ⚠️  ロールの検証に失敗: {e}")
    else:
        print("\n  ℹ️  --role-arn を指定すると、デプロイ済みロールの実効権限も検証できます。")

    print("\n  ✅ IAM ポリシー検証完了")


if __name__ == "__main__":
    main()
