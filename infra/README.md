# デモ環境インフラストラクチャ

## 概要

Session Manager 経由で接続する EC2 デモ環境。SSH キー不要。
ハンズオン資材は S3 バケットに保管し、EC2 起動時に自動ダウンロード。
EC2 からは SageMaker 実行ロールを `PassRole` して、SageMaker のトレーニングジョブ・エンドポイント・処理ジョブを操作します。

## 初回セットアップ

リポジトリをクローンした後、以下を 1 回実行してください：

```bash
git config core.hooksPath .githooks
```

これにより `git push` 時に `handson/` フォルダに変更があれば自動で S3 にアップロードされます。

## デプロイ手順

### Step 1: 資材を S3 にアップロード

```bash
chmod +x infra/upload-assets.sh
./infra/upload-assets.sh
```

### Step 2: CloudFormation スタックの作成

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws cloudformation create-stack \
  --stack-name handson-mlengineering-demo-env \
  --template-body file://infra/demo-ec2.yaml \
  --parameters \
    ParameterKey=AssetsBucket,ParameterValue=handson-mlengineering-assets-$ACCOUNT_ID \
    ParameterKey=AssetsPrefix,ParameterValue=handson-assets \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

### Step 3: 完了を待機

```bash
aws cloudformation wait stack-create-complete --stack-name handson-mlengineering-demo-env --region us-east-1
```

### Step 4: Session Manager で接続

```bash
# インスタンス ID を取得
INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name handson-mlengineering-demo-env \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
  --output text)

# 接続
aws ssm start-session --target $INSTANCE_ID

# 接続後
cd ~/handson
echo $SAGEMAKER_ROLE_ARN   # SageMaker 実行ロールが設定済み
```

### Step 5: SageMaker ドメイン / Studio について

多くのハンズオンは EC2 上の SageMaker Python SDK からリモートジョブを起動する形式のため、
SageMaker Studio ドメインは必須ではありません。Studio を使ったラボ（ラボ 1〜7）を実施する場合は、
AWS Builder Labs 環境、または別途 SageMaker ドメインを作成してください。

## 運用フロー

```
初回:   upload-assets.sh → CFn create-stack
前日:   aws ec2 start-instances --instance-ids <ID>
当日:   aws ssm start-session → cd ~/handson
夜間:   23:00 JST に自動停止（Lambda）
更新時: upload-assets.sh → EC2内で再取得
```

## 資材の更新方法

ハンズオン内容を更新した場合:

```bash
# 1. S3 に最新版をアップロード（起動中なら EC2 内も自動更新）
./infra/upload-assets.sh

# 2. 手動で EC2 内を更新する場合
aws ssm start-session --target <INSTANCE_ID>
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws s3 cp s3://handson-mlengineering-assets-$ACCOUNT_ID/handson-assets/handson.tar.gz /tmp/
rm -rf ~/handson
mkdir ~/handson
tar -xzf /tmp/handson.tar.gz -C ~/handson --strip-components=1
```

## コスト見積もり

| リソース | 月間コスト（研修時のみ起動） |
|---------|--------------------------|
| EC2 t3.large（1日8時間×5日） | ~$15 |
| EBS 50GB gp3 | ~$4 |
| S3（資材保管） | < $0.10 |
| Lambda（自動停止） | < $0.01 |
| SageMaker（トレーニング / エンドポイント デモ実行） | $3-10/研修回 |
| **合計** | **~$25-35/月** |

> ℹ️ SageMaker のリアルタイムエンドポイントは起動している間課金されます。各ハンズオンの最後に必ず削除するか、`handson/cleanup_all.sh` を実行してください。

## クリーンアップ

```bash
# スタック削除
aws cloudformation delete-stack --stack-name handson-mlengineering-demo-env --region us-east-1

# S3 バケット削除
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws s3 rb s3://handson-mlengineering-assets-$ACCOUNT_ID --force
```
