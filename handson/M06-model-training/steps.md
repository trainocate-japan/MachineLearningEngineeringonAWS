# モジュール 6: ML モデルのトレーニング - ハンズオン手順

## 前提: 特徴量データが生成済みであること

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000
python M04-feature-engineering/data_cleaning.py
python M04-feature-engineering/feature_transform.py
cd ~/handson/M06-model-training
```

---

## パート 1: ローカル XGBoost トレーニング（20分）

### ステップ 1.1: ローカルトレーニングの実行

```bash
python local_train.py
```

学習曲線（`output/learning_curve.png`）を署名付き URL で確認する場合：

```bash
python local_train.py --publish
```

### ステップ 1.2: モデルの構成要素を確認

- **モデル = アルゴリズム + 重み（パラメータ）**
- **損失関数**: `binary:logistic` の logloss を最小化
- **最適化**: 勾配ブースティング（各ラウンドで誤差を縮小）
- **ハイパーパラメータ**: `max_depth`, `eta`（学習率）, `num_round`（ツリー数）

### ステップ 1.3: 学習曲線の観察（過学習 vs 過小適合）

`output/learning_curve.png` で train と validation の logloss を比較します：

| 状態 | 兆候 |
|------|------|
| 理想的フィッティング | 両者が低く、乖離が小さい |
| 過学習（高分散） | train は下がるが validation が上がり始める |
| 過小適合（高バイアス） | 両者とも高止まり |

### ステップ 1.4: 特徴量重要度

出力の「特徴量重要度 上位8」を確認します。`education_ordinal` や `age_std` など、EDA（M03）で見た相関と整合するはずです。

> 💡 ハイパーパラメータを変えて（例: `max_depth=10`, `num_round=500`）再実行し、過学習が起きる様子を観察するのも有効です。

---

## パート 2: SageMaker トレーニングジョブ（20分）

### ステップ 2.1: 構成の確認（dry-run）

```bash
python sagemaker_training.py --dry-run
```

### ステップ 2.2: トレーニングジョブの実行

```bash
python sagemaker_training.py
```

処理の流れ：
1. 特徴量データを train/validation に分割し、ヘッダなし CSV で S3 にアップロード
2. 組み込み XGBoost Estimator（`ml.m5.large`）を構成
3. `fit()` でマネージドトレーニングジョブを実行
4. モデルアーティファクト（`model.tar.gz`）が S3 に保存される

> ⏱️ 初回はコンテナ起動のため 3〜5 分かかります。
> 完了後、`estimator.model_data`（S3 パス）とジョブ名が表示されます。

### ステップ 2.3: コンピュートオプションのディスカッション

| インスタンス | 用途 |
|-------------|------|
| CPU | 小規模モデル・前処理・特徴量エンジニアリング（本ハンズオン） |
| GPU | 行列演算・ディープラーニング |
| AWS Trainium | 大規模分散トレーニング・コスト効率 |

---

## パート 3: まとめ（5分）

- モデルの構成要素（アルゴリズム + 重み・損失・最適化・ハイパーパラメータ）
- 学習曲線によるフィッティング状態の判断
- ローカル → SageMaker トレーニングジョブへのスケール

次のモジュール（M07）では、このモデルを評価し、ハイパーパラメータチューニングで改善します。

---

## クリーンアップ

トレーニングジョブは完了で自動停止します。学習出力の S3 は `cleanup_all.sh` で削除されます。

---

## 参考ドキュメント

- [SageMaker 組み込み XGBoost](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)
- [XGBoost ハイパーパラメータ](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html)
