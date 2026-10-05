# モジュール 7: モデル評価とハイパーパラメータチューニング - ハンズオン手順

## 前提: 特徴量データが生成済みであること

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000
python M04-feature-engineering/data_cleaning.py
python M04-feature-engineering/feature_transform.py
cd ~/handson/M07-evaluation-tuning
```

---

## パート 1: モデル評価指標（15分）

### ステップ 1.1: 評価の実行

```bash
python evaluate_model.py
```

出力される混同行列と各指標を確認します：
- 混同行列（TN / FP / FN / TP）
- Accuracy / Precision / Recall / F1 / AUC
- `output/confusion_matrix.png` と `output/roc_curve.png`

### ステップ 1.2: 指標の使い分けディスカッション

| 指標 | 定義 | 重視すべき場面 |
|------|------|--------------|
| Precision（適合率） | TP/(TP+FP) | 誤検知(FP)のコストが高い |
| Recall（再現率） | TP/(TP+FN) | 見逃し(FN)のコストが高い（疾病・不正検出） |
| F1 | Precision と Recall の調和平均 | バランス重視 |
| AUC | ROC 曲線下面積 | 閾値非依存の総合性能（0.5=ランダム, 1.0=完璧） |

**議論**: 収入予測でマーケティング配信先を選ぶ場合、Precision と Recall のどちらを重視すべきでしょうか？（配信コスト vs 機会損失）

---

## パート 2: バイアスと分散のトレードオフ（15分）

### ステップ 2.1: デモの実行

```bash
python bias_variance.py
```

`max_depth` と `num_round` を段階的に増やし、train loss と test loss の推移を観察します。
`output/bias_variance.png` を確認してください。

### ステップ 2.2: 観察

| 複雑さ | 状態 | 兆候 |
|-------|------|------|
| 低 | 過小適合（高バイアス） | train/test ともに損失が高い |
| 中 | 理想 | test loss が最小 |
| 高 | 過学習（高分散） | train は低いが test が下げ止まり/上昇 |

**補足**:
- 高バイアスの原因: モデルが単純すぎる / 不適切な特徴量
- 高分散の原因: モデルが複雑すぎる / 無関係な情報が多い / 学習しすぎ

→ 理想は「バイアス低・分散低」。これを次のチューニングで探します。

---

## パート 3: SageMaker 自動モデルチューニング（AMT）（15分）

### ステップ 3.1: 構成とチューニング手法の確認（dry-run）

```bash
python hyperparameter_tuning.py --dry-run
```

探索範囲とチューニング手法の比較表を確認します。

| 手法 | 特徴 |
|------|------|
| グリッド検索 | 全組み合わせ・網羅的・高コスト |
| ランダム検索 | ランダム・効率的・並列向き |
| ベイズ最適化 | 過去結果を活用・少試行で高精度（本デモ） |
| Hyperband | 有望構成にリソース集中・反復アルゴリズム向き |

### ステップ 3.2: チューニングジョブの実行

```bash
python hyperparameter_tuning.py --max-jobs 8 --max-parallel 2
```

- ベイズ最適化で `validation:auc` を最大化
- 探索対象: `max_depth`, `eta`, `min_child_weight`, `subsample`
- 完了後、最良ジョブのハイパーパラメータを表示

> ⏱️ 複数ジョブを実行するため 10〜20 分かかります。
> デモ時間が限られる場合は `--max-jobs 4` に減らしてください。

### ステップ 3.3: 結果の確認

- SageMaker コンソール → トレーニング → ハイパーパラメータチューニングジョブ
- 各トライアルの AUC と選ばれたハイパーパラメータを比較

---

## パート 4: まとめ（5分）

- 分類指標（混同行列・Precision・Recall・F1・ROC/AUC）と使い分け
- バイアス・分散のトレードオフと過学習・過小適合
- チューニング手法の違い
- SageMaker AMT による自動探索

次のモジュール（M08）では、チューニングしたモデルをデプロイします。

---

## 参考ドキュメント

- [SageMaker 自動モデルチューニング](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
- [チューニング戦略（ベイズ最適化 / Hyperband）](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html)
- [分類指標（scikit-learn）](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [XGBoost ハイパーパラメータ](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html)
