# モジュール 3: データ処理と探索的データ分析 - ハンズオン手順

## パート 0: データセットの準備（5分）

### ステップ 0.1: 共通データセットを生成する

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000
```

以下の 2 ファイルが `common/data/` に生成されます：
- `clean_income.csv` : クリーンなトレーニング用データ
- `raw_income.csv` : 品質問題を含む生データ（EDA 用）

---

## パート 1: 探索的データ分析（EDA）（15分）

### ステップ 1.1: データ形式の復習（ディスカッション）

データ形式を整理します：

| 形式 | ファイルタイプ | 特徴 |
|------|--------------|------|
| 行ベース | CSV / Avro / RecordIO | 行単位の密なアクセスに適する |
| 列ベース | Parquet / ORC | 列単位の集約・スパースデータに適する |
| オブジェクト表記 | JSON / JSONL | 柔軟・階層的な構造 |

**議論**: 今回の収入予測データ（表形式・列ごとの集約が多い）には、どの形式が適しているでしょうか？

### ステップ 1.2: EDA の実行

```bash
cd ~/handson/M03-data-processing
python explore_data.py
```

出力を確認し、以下を議論します：
- 数値列の記述統計（`describe()`）で `age` の max に注目
- 欠損値がどの列にどれだけあるか
- 重複行の件数
- IQR 法で検出された外れ値
- `income_text` の書式ゆれ

**観察ポイント**: 「不正確・重複・欠落データ」が実データでどう現れるかを確認します。

---

## パート 2: データ可視化（15分）

### ステップ 2.1: 可視化の実行

```bash
python visualize_data.py
```

`output/` に 5 つの PNG が生成されます。

### ステップ 2.2: 可視化手法の目的別整理

| 目的 | グラフ | 生成ファイル |
|------|-------|-------------|
| 分布 | ヒストグラム | `hist_age.png` |
| 分布 | ボックスプロット | `box_income.png` |
| 関係 | 散布図 | `scatter_edu_income.png` |
| 関係 | 相関ヒートマップ | `heatmap_corr.png` |
| 構成/比較 | 棒グラフ | `bar_education.png` |

### ステップ 2.3: 考察

- `education_num` と `income_high` の相関（ヒートマップ）
- 高収入層が収入ボックスプロットの外れ値側に分布
- 学歴が上がるほど高収入割合が増える（棒グラフ）

→ これらの気づきが、M04 の特徴量エンジニアリングの方針につながります。

---

## パート 3: 生成した図をブラウザで確認する（10分）

EC2 は Session Manager 専用でインバウンドを遮断しているため、生成した PNG は
S3 にアップロードして**署名付き URL**でブラウザから確認します（公開バケットや
ポート開放は不要）。

### ステップ 3.1: 可視化と同時に URL を発行する

```bash
python visualize_data.py --publish
```

各 PNG の署名付き URL（15 分有効）が表示されます。ブラウザで開くと図を確認できます。

### ステップ 3.2: 既存の画像を個別に発行する

```bash
# 単一ファイル
python ../common/publish.py output/heatmap_corr.png

# 複数ファイルまとめて
python ../common/publish.py output/*.png
```

> 💡 URL の有効期限は最大 15 分です。期限切れになったら再度コマンドを実行して発行し直してください。
> 画像は SageMaker デフォルトバケットの `mle-handson/published/` に保管されます。

---

## パート 4: まとめ（5分）

- データ形式・データ型の理解
- EDA による品質問題の特定（欠損・重複・外れ値・書式ゆれ）
- 目的に応じた可視化手法の選択
- 署名付き URL による図の共有

次のモジュール（M04）では、ここで特定したデータ品質問題を実際に修正し、
特徴量エンジニアリングを行います。

---

## 参考ドキュメント

- [Amazon S3 署名付き URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html)
- [Amazon SageMaker でのデータ準備](https://docs.aws.amazon.com/sagemaker/latest/dg/data-prep.html)
- [pandas ユーザーガイド](https://pandas.pydata.org/docs/user_guide/index.html)
