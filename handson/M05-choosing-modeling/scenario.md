# モジュール 5: モデリングアプローチの選択 - ハンズオンシナリオ

## シナリオ概要

特徴量が準備できたので、どのアルゴリズムでモデルを作るかを決めます。あなたは AnyCompany の ML エンジニアとして、「解釈しやすく高速な線形モデル」と「高精度になりやすい勾配ブースティング」のどちらが今回の収入予測に適しているかを、実際に SageMaker で両方を学習させて比較・判断します。

## 学習目標

このハンズオンを完了すると、以下ができるようになります：

1. **アルゴリズムを実際に比較する**: Linear Learner と XGBoost を同じデータで学習させ、検証スコアと学習時間を比較する
2. **選択基準を理解する**: 問題タイプ・データ特性・解釈可能性・コストのトレードオフを判断する
3. **組み込みアルゴリズムの使い方**: SageMaker 組み込みアルゴリズムのコンテナ URI 取得とハイパーパラメータ設定を体験する

## データセット

M04 で生成した `features_income.csv`（ターゲット `income_high`）を使用します。

## 使用する AWS サービス / ツール

- Amazon SageMaker 組み込み Linear Learner
- Amazon SageMaker 組み込み XGBoost
- Amazon SageMaker Training（トレーニングジョブ）
- Amazon S3

## 所要時間

約 45 分

## 前提条件

- M04 を完了していること（特徴量データが生成済み）
- Python パッケージ: `numpy`, `pandas`, `scikit-learn`, `sagemaker`, `boto3`
- `$SAGEMAKER_ROLE_ARN` が設定済み
