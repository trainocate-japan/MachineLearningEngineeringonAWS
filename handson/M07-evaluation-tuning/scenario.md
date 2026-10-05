# モジュール 7: モデル評価とハイパーパラメータチューニング - ハンズオンシナリオ

## シナリオ概要

M06 でトレーニングしたモデルが「どれくらい良いのか」を定量的に評価します。あなたは AnyCompany の ML エンジニアとして、混同行列・適合率・再現率・F1・ROC/AUC といった指標でモデルを評価し、バイアスと分散のトレードオフを観察します。その後、Amazon SageMaker 自動モデルチューニング（AMT）で最適なハイパーパラメータを探索します。

## 学習目標

このハンズオンを完了すると、以下ができるようになります：

1. **評価指標を理解する**: 混同行列、Accuracy、Precision、Recall、F1、ROC 曲線と AUC
2. **指標の使い分け**: クラス不均衡や誤分類コストに応じて適切な指標を選ぶ
3. **バイアス・分散のトレードオフ**: モデルの複雑さを変えて過学習・過小適合を観察する
4. **ハイパーパラメータチューニング手法**: グリッド検索・ランダム検索・ベイズ最適化・Hyperband の違いを理解する
5. **SageMaker AMT**: 自動モデルチューニングジョブで最適なハイパーパラメータを探索する

## データセット

M04 で生成した `features_income.csv`（ターゲット `income_high`）を使用します。

## 使用する AWS サービス / ツール

- Amazon SageMaker 自動モデルチューニング（AMT / HyperparameterTuner）
- Amazon SageMaker 組み込み XGBoost
- scikit-learn（評価指標・学習曲線）
- xgboost

## 所要時間

約 45 分

## 前提条件

- M04 を完了していること（特徴量データが生成済み）
- Python パッケージ: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `sagemaker`, `boto3`
- `$SAGEMAKER_ROLE_ARN` が設定済み（AMT を実行する場合）
