# モジュール 10: MLOps と自動デプロイ - ハンズオンシナリオ

## シナリオ概要

手作業でのトレーニング・デプロイは再現性に欠け、ミスも起きやすいものです。あなたは AnyCompany の ML エンジニアとして、データ処理 → トレーニング → 評価 → モデル登録までを自動化する SageMaker Pipeline を構築し、Model Registry でモデルをバージョン管理・承認します。

## 学習目標

このハンズオンを完了すると、以下ができるようになります：

1. **MLOps の基本**: DevOps との違い（データ・モデル・ワークフローのバージョン管理）を理解する
2. **SageMaker Pipelines**: 前処理 → 学習 → 評価 → 条件付き登録を 1 つのパイプラインで自動化する
3. **Model Registry**: モデルをバージョン管理し、承認ワークフローを運用する

## データセット

M03〜M09 と同じ「成人の収入予測」データセットを使用します。パイプラインは生データ
`raw_income.csv` を入力とし、M04 の `processing_entry.py` を前処理ステップとして再利用します。

## 使用する AWS サービス / ツール

- Amazon SageMaker Pipelines（ワークフローオーケストレーション）
- Amazon SageMaker Model Registry（Model Package Group）
- Amazon SageMaker Processing / Training

## 所要時間

約 45 分

## 前提条件

- データセットが生成済み（`common/generate_dataset.py`）
- Python パッケージ: `numpy`, `pandas`, `scikit-learn`, `sagemaker`, `boto3`
- `$SAGEMAKER_ROLE_ARN` が設定済み
