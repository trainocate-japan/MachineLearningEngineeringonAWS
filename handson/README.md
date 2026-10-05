# Machine Learning Engineering on AWS - ハンズオンガイド

## コース概要

このハンズオンガイドは、Amazon SageMaker AI を中心とした機械学習エンジニアリングの実践的なシナリオと手順を提供します。各ハンズオンで作成したリソースやソリューションは、デモとして見せることができます。

ハンズオンは ML ライフサイクル全体（データ処理 → 特徴量エンジニアリング → モデルトレーニング → 評価・チューニング → デプロイ → MLOps → モニタリング）をカバーします。

## ハンズオンモジュール一覧

モジュール番号はコースのモジュール（Mod03〜Mod11）に対応しています。
Mod00/01/02/12 は座学中心のため、ハンズオン素材はありません。

| モジュール | テーマ | ハンズオン時間 |
|-----------|--------|--------------|
| M03 | データ処理と探索的データ分析（EDA・可視化・署名付き URL） | 45分 |
| M04 | データ変換と特徴量エンジニアリング（SageMaker Processing） | 45分 |
| M05 | モデリングアプローチの選択（Linear vs XGBoost 比較） | 45分 |
| M06 | ML モデルのトレーニング（XGBoost トレーニングジョブ） | 45分 |
| M07 | モデルの評価とチューニング（AMT） | 45分 |
| M08 | モデルデプロイ戦略（リアルタイム / バッチ / トラフィックシフト） | 60分 |
| M09 | AWS ML リソースの保護（IAM Policy Simulator） | 30分 |
| M10 | MLOps と自動デプロイ（SageMaker Pipelines・Model Registry） | 45分 |
| M11 | モデルモニタリングとドリフト検出（SageMaker Model Monitor） | 45分 |

## 前提条件

### 環境要件
- AWS アカウント（管理者アクセス）
- AWS CLI v2 設定済み
- Python 3.12+（EC2 環境では venv が自動で有効化されます）
- SageMaker 実行ロール（EC2 環境では `$SAGEMAKER_ROLE_ARN` に自動設定されます）

### Python パッケージ
- `boto3`
- `sagemaker`（SageMaker Python SDK）
- `numpy` / `pandas` / `scikit-learn`
- `matplotlib` / `seaborn`
- `xgboost` / `pyarrow`

### リージョン
- 推奨: `us-east-1`（バージニア北部）または `us-west-2`（オレゴン）
- SageMaker AI の全機能が利用可能なリージョンを選択してください

## フォルダ構造

```
handson/
├── README.md                           # このファイル
├── cleanup_all.sh                      # 全リソース一括削除スクリプト
├── common/
│   ├── generate_dataset.py             # 共通サンプルデータ生成（収入予測データセット）
│   └── publish.py                      # 生成画像を S3 にアップし署名付き URL を発行
├── M03-data-processing/                # データ処理と EDA
│   ├── scenario.md
│   ├── steps.md
│   ├── explore_data.py                 # pandas による EDA
│   └── visualize_data.py               # 可視化（--publish で署名付き URL 発行）
├── M04-feature-engineering/            # データ変換と特徴量エンジニアリング
│   ├── scenario.md
│   ├── steps.md
│   ├── data_cleaning.py                # クリーニング（欠損・重複・外れ値）
│   ├── feature_transform.py            # エンコーディング・スケーリング・ビニング
│   ├── pca_demo.py                     # PCA による次元削減
│   ├── sagemaker_processing.py         # SageMaker Processing ジョブ起動
│   └── processing_entry.py             # Processing コンテナ内の前処理スクリプト
├── M05-choosing-modeling/              # モデリングアプローチの選択
│   ├── scenario.md
│   ├── steps.md
│   └── compare_algorithms.py           # Linear Learner vs XGBoost を実学習して比較
├── M06-model-training/                 # モデルトレーニング
│   ├── scenario.md
│   ├── steps.md
│   ├── local_train.py                  # ローカル XGBoost 学習（--publish 対応）
│   └── sagemaker_training.py           # SageMaker トレーニングジョブ
├── M07-evaluation-tuning/              # 評価とチューニング
│   ├── scenario.md
│   ├── steps.md
│   ├── evaluate_model.py               # 混同行列・精度・再現率・F1・ROC/AUC（--publish 対応）
│   ├── bias_variance.py                # バイアス・分散のトレードオフ（--publish 対応）
│   └── hyperparameter_tuning.py        # SageMaker 自動モデルチューニング（AMT）
├── M08-deployment/                     # デプロイ戦略
│   ├── scenario.md
│   ├── steps.md
│   ├── _common.py                      # 学習済みモデル準備の共通ヘルパー
│   ├── realtime_endpoint.py            # リアルタイムエンドポイント
│   ├── batch_transform.py              # バッチ変換
│   └── traffic_shifting.py             # ブルー/グリーン（Linear/Canary）
├── M09-securing-resources/             # ML リソースの保護
│   ├── scenario.md
│   ├── steps.md
│   └── verify_iam_policy.py            # IAM Policy Simulator で許可/拒否を実検証
├── M10-mlops/                          # MLOps と自動デプロイ
│   ├── scenario.md
│   ├── steps.md
│   ├── sagemaker_pipeline.py           # SageMaker Pipelines（前処理→学習→評価→登録）
│   ├── evaluation_entry.py             # Pipeline 評価ステップのスクリプト
│   └── model_registry.py               # Model Registry 登録・承認
├── M11-monitoring/                     # モニタリング
│   ├── scenario.md
│   ├── steps.md
│   ├── data_capture.py                 # データキャプチャ設定
│   ├── model_monitor.py                # ベースライン作成 + モニタリングスケジュール
│   └── drift_simulation.py             # ドリフトのシミュレーションと検出
```

## 使い方

1. 各モジュールフォルダ内の `scenario.md` でシナリオと学習目標を確認
2. `steps.md` の手順に従ってハンズオンを実施
3. Python スクリプトを実行してデモ動作を確認
4. 終了後は下記クリーンアップ手順に従ってリソースを削除

> 💡 まず `common/generate_dataset.py` を実行して共通のサンプルデータセットを生成してください。M03〜M11 の多くがこのデータセット（成人の収入予測）を使用します。
>
> 💡 EC2 は Session Manager 専用（インバウンド遮断）のため、生成した画像は `common/publish.py` または各スクリプトの `--publish` で署名付き URL を発行し、ブラウザで確認します。

## クリーンアップ

全モジュールで作成したリソースを一括で削除するスクリプトを用意しています。

### EC2 ハンズオン環境での実行

```bash
cd ~/handson
bash cleanup_all.sh
```

### 対象リソース一覧

| モジュール | 削除対象 |
|-----------|---------|
| M03 | ローカル生成物・公開画像（S3 の published 配下） |
| M04 | SageMaker Processing ジョブ出力（S3） |
| M05 | トレーニングジョブ出力（S3） |
| M06 | SageMaker トレーニングジョブ出力（S3）、Model |
| M07 | ハイパーパラメータチューニングジョブ、Model |
| M08 | **リアルタイムエンドポイント**、エンドポイント設定、Model、バッチ変換出力 |
| M09 | なし（IAM シミュレーションのみ・リソース作成なし） |
| M10 | SageMaker Pipeline、Model Package Group |
| M11 | モニタリングスケジュール、エンドポイント、ベースライン出力（S3） |

### 注意事項

- スクリプトは冪等です（リソースが存在しなければスキップします）
- 実行前に `aws sts get-caller-identity` で正しいアカウントか確認してください
- **リアルタイムエンドポイントは起動している間ずっと課金されます。** 必ず削除してください。

## コスト管理

- 各ハンズオンの推定コスト: $1〜$5
- 使用後は必ずリソースを削除してください（特に SageMaker エンドポイント）
- トレーニングジョブ・処理ジョブは実行時間分のみ課金されます

## トラブルシューティング

### SageMaker 実行ロールが見つからない
```
ValueError: Couldn't call 'get_role' to get Role ARN
```
→ EC2 環境では `$SAGEMAKER_ROLE_ARN` を使用します。スクリプトは環境変数を優先的に読み込みます。
→ ローカル環境では `--role-arn` 引数、または `SAGEMAKER_ROLE_ARN` を手動で設定してください。

### リージョン未対応 / インスタンスタイプエラー
```
ResourceLimitExceeded
```
→ `us-east-1` または `us-west-2` に変更、またはサービスクォータの引き上げを申請してください。
→ ハンズオンでは `ml.m5.large` / `ml.m5.xlarge` を使用します。

### S3 アクセスエラー
→ SageMaker デフォルトバケット（`sagemaker-<region>-<account-id>`）への権限を確認してください。
