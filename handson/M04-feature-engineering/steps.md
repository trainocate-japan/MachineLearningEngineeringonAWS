# モジュール 4: データ変換と特徴量エンジニアリング - ハンズオン手順

## 前提: データセットが生成済みであること

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000   # M03 で未実施の場合
cd ~/handson/M04-feature-engineering
```

---

## パート 1: データクリーニング（10分）

### ステップ 1.1: クリーニングの実行

```bash
python data_cleaning.py
```

以下の処理を確認します：
- **書式ゆれ**: `income_text`（`"73K"`, `"$93605"`, `"117,451"`）→ 数値 `income_usd`
- **重複排除**: 完全重複行を削除
- **外れ値**: `age>=120` を欠損扱いにして中央値補完
- **欠損補完**: `age`/`income_usd` は中央値、`state` は最頻値

出力: `../common/data/cleaned_income.csv`

### ステップ 1.2: 補完手法のディスカッション

| 欠損の種類 | 説明 | 対処 |
|-----------|------|------|
| MCAR | 完全にランダムな欠損 | 行削除 or 補完 |
| MAR | 他の変数に依存する欠損 | 補完（回帰補完等） |
| MNAR | 欠損自体に意味がある | 慎重に扱う |

**議論**: なぜ `age` の補完に平均値ではなく中央値を使ったのでしょうか？（外れ値の影響）

---

## パート 2: 特徴量変換（15分）

### ステップ 2.1: 変換の実行

```bash
python feature_transform.py
```

### ステップ 2.2: エンコーディング手法の使い分け

| 手法 | 対象列 | 理由 |
|------|-------|------|
| 序数エンコーディング | `education` | 学歴に順序がある |
| ワンホットエンコーディング | `workclass`, `occupation`, `state` | 順序のない名義変数 |

### ステップ 2.3: 数値特徴量エンジニアリング

| 手法 | 対象 | 効果 |
|------|------|------|
| 標準化 | `age`, `hours_per_week` | 平均0・標準偏差1、外れ値の影響軽減 |
| 正規化 | `education_num` | 0〜1 スケール |
| 対数変換 | `income_usd` | 右に歪んだ分布を圧縮 |
| ビニング | `age` | 世代カテゴリに離散化 |

出力: `../common/data/features_income.csv`（先頭列が `income_high`）

---

## パート 3: 特徴量選択と PCA（10分）

### ステップ 3.1: PCA デモの実行

```bash
python pca_demo.py
```

数値特徴量に PCA を適用し、各主成分の説明分散比を確認します。

### ステップ 3.2: 考察

- 相関の高い特徴量が少数の主成分に集約される
- 分散の 90% を説明するのに必要な主成分数を確認
- **トレードオフ**: 次元削減 vs 解釈可能性

**補足**: 住宅データの「面積・寝室・バスルーム」が「サイズ」という主成分に集約されるのと同じ原理です。

---

## パート 4: SageMaker Processing でスケール実行（10分）

### ステップ 4.1: 構成の確認（dry-run）

```bash
python sagemaker_processing.py --dry-run
```

ジョブ構成（入力・エントリースクリプト・出力・分割比率）を確認します。

### ステップ 4.2: Processing ジョブの実行

```bash
python sagemaker_processing.py
```

処理の流れ：
1. 生データを S3 にアップロード
2. `SKLearnProcessor` が `processing_entry.py` をマネージド環境で実行
3. `train.csv` / `validation.csv` / `test.csv` を S3 に出力（70/15/15 に層化分割）

> ⏱️ 初回はコンテナ起動のため数分かかります。
> 出力 S3 パスは M06（モデルトレーニング）の入力として利用します。

### ステップ 4.3: ローカル処理との違い

- **ローカル**: 小規模データの試行錯誤に向く
- **SageMaker Processing**: 大規模データ・再現性・自動ワークフロー統合に向く（分散処理対応）

---

## パート 5: まとめ（5分）

- データクリーニングの 4 手法（書式・重複・外れ値・欠損）
- エンコーディングの使い分け（序数 vs ワンホット）
- 数値変換（標準化・正規化・対数・ビニング）
- 特徴量選択と PCA
- ローカル処理をマネージドジョブにスケール

---

## 参考ドキュメント

- [SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [SKLearnProcessor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html)
- [scikit-learn 前処理](https://scikit-learn.org/stable/modules/preprocessing.html)
- [主成分分析（PCA）](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
