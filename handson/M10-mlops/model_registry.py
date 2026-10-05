"""
モジュール 10: SageMaker Model Registry

Model Package Group に登録されたモデルを一覧し、
承認ステータス（PendingManualApproval → Approved）を変更します。

扱う内容:
- SageMaker Model Registry によるモデルバージョニング
  （バージョン・グループ・承認ワークフロー・デプロイ自動化・ガバナンス）
- SageMaker Projects（CI/CD での Model Registry 活用）

sagemaker_pipeline.py を実行すると mle-handson-model-group にモデルが登録されます。
このスクリプトはそのモデルパッケージを一覧し、最新を承認します。

--dry-run で AWS を呼ばず説明のみ表示。
--approve で最新の承認待ちモデルを Approved に変更。
"""

import argparse

MODEL_GROUP = "mle-handson-model-group"


def main():
    parser = argparse.ArgumentParser(description="Model Registry の操作")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve", action="store_true",
                        help="最新の承認待ちモデルを Approved に変更")
    args = parser.parse_args()

    print("=" * 60)
    print(" SageMaker Model Registry")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY-RUN] Model Registry の操作:")
        print(f"  - 対象グループ: {MODEL_GROUP}")
        print("  - list_model_packages でバージョン一覧を取得")
        print("  - update_model_package で承認ステータスを変更")
        print("\n  承認ワークフロー:")
        print("    PendingManualApproval → (レビュー) → Approved / Rejected")
        print("    Approved になったモデルを CI/CD が自動デプロイ可能")
        return

    import boto3

    sm = boto3.client("sagemaker")

    # グループの存在確認
    try:
        sm.describe_model_package_group(ModelPackageGroupName=MODEL_GROUP)
    except sm.exceptions.ClientError:
        print(f"\n  ⚠️  Model Package Group が見つかりません: {MODEL_GROUP}")
        print("  先に sagemaker_pipeline.py を実行してモデルを登録してください。")
        return

    resp = sm.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    packages = resp.get("ModelPackageSummaryList", [])

    if not packages:
        print(f"\n  登録済みモデルがありません（グループ: {MODEL_GROUP}）")
        print("  先に sagemaker_pipeline.py を実行してください。")
        return

    print(f"\n  [登録済みモデルパッケージ: {MODEL_GROUP}]")
    for p in packages:
        print(f"    v{p['ModelPackageVersion']}: {p['ModelApprovalStatus']}"
              f"  ({p['CreationTime']:%Y-%m-%d %H:%M})")

    if args.approve:
        latest = packages[0]
        arn = latest["ModelPackageArn"]
        if latest["ModelApprovalStatus"] == "Approved":
            print(f"\n  最新 v{latest['ModelPackageVersion']} は既に Approved です。")
            return
        sm.update_model_package(
            ModelPackageArn=arn,
            ModelApprovalStatus="Approved",
        )
        print(f"\n  ✅ v{latest['ModelPackageVersion']} を Approved に変更しました。")
        print("     本番デプロイパイプラインのトリガーになり得ます。")
    else:
        print("\n  ℹ️  承認するには --approve を付けて再実行してください。")


if __name__ == "__main__":
    main()
