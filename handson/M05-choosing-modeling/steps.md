# モジュール 5: モデリングアプローチの選択 - ハンズオン手順

## 前提: 特徴量データが生成済みであること

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000
python M04-feature-engineering/data_cleaning.py
python M04-feature-engineering/feature_transform.py
cd ~/handson/M05-choosing-modeling
```

---

## パート 1: 選択基準の整理（10分）

アルゴリズム選択で考慮する観点を整理します：

| 考慮事項 | 内容 |
|---------|------|
| 問題タイプ | 分類 / 回帰 / クラスタリング / 異常検出 など |
| データ特性 | 構造化（CPU）vs 非構造化（GPU・分散） |
| 解釈可能性 | 規制・高リスクドメインで重要 |
| コスト | 学習時間・インスタンス種別 |

**アルゴリズム選択の例**:
- リアルタイム不正検出 → Random Cut Forest
- 解釈性が必要な再入院予測 → ロジスティック回帰 / Linear Learner
- ラベル付きの構造化データ分類 → Linear Learner / XGBoost

→ 今回は収入予測（構造化データの二項分類）なので、**Linear Learner** と **XGBoost** を実際に比較します。

---

## パート 2: アルゴリズムの比較（25分）

### ステップ 2.1: 構成の確認（dry-run）

```bash
python compare_algorithms.py --dry-run
```

### ステップ 2.2: 両アルゴリズムを学習して比較

```bash
python compare_algorithms.py
```

処理の流れ：
1. 特徴量データを train/validation に分割し S3 にアップロード
2. Linear Learner と XGBoost をそれぞれ SageMaker トレーニングジョブで学習
3. 検証スコアと学習時間を比較表で表示

> ⏱️ 2 つのジョブを順番に実行するため 5〜10 分かかります。
> 片方だけ試すには `--algo linear` または `--algo xgboost` を指定します。

### ステップ 2.3: 結果の考察

| アルゴリズム | 特徴 |
|-------------|------|
| Linear Learner | 高速・解釈しやすい・ベースライン / 規制対応向き |
| XGBoost | 非線形パターンを捉え高精度になりやすい・解釈性は下がる |

**議論**: スコアが拮抗する場合、どちらを本番採用すべきでしょうか？（解釈性・学習コスト・運用のしやすさ）

---

## パート 3: まとめ（10分）

- 同じデータで複数アルゴリズムを実際に回して比較する実践的な進め方
- 解釈可能性 vs 精度、学習時間・コストのトレードオフ
- 構造化データの二項分類では Linear / XGBoost がまず候補になる

次のモジュール（M06）では、選択した XGBoost を本格的にトレーニングします。

---

## クリーンアップ

このモジュールはトレーニングジョブのみ（完了で自動停止）で、常時課金されるリソースは
作りません。学習出力の S3 は `cleanup_all.sh` で削除されます。

---

## 参考ドキュメント

- [SageMaker 組み込みアルゴリズム一覧](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
- [Linear Learner アルゴリズム](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)
- [XGBoost アルゴリズム](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
