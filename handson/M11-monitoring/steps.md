# モジュール 11: モデルモニタリングとドリフト検出 - ハンズオン手順

## 前提: 特徴量データが生成済みであること

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000
python M04-feature-engineering/data_cleaning.py
python M04-feature-engineering/feature_transform.py
cd ~/handson/M11-monitoring
```

> ⚠️ **コスト注意**: このモジュールはリアルタイムエンドポイントとモニタリングスケジュールを作成します。
> 終了時に必ず `cleanup_all.sh` を実行してください。

---

## パート 1: ドリフトの種類（10分）

### ステップ 1.1: ドリフトシミュレーションの実行（ローカル）

```bash
python drift_simulation.py
```

トレーニングデータからベースライン統計を作り、本番データの分布を意図的にシフトさせて、ベースラインからの逸脱を検出するロジックを体験します。これは Model Monitor の内部動作の簡略版です。

### ステップ 1.2: ドリフトの種類

| ドリフト | 内容 |
|---------|------|
| データ品質ドリフト | 入力データの統計/スキーマが変化（本デモ） |
| モデル品質ドリフト | 予測精度が時間とともに劣化 |
| バイアスドリフト | 特定グループへの予測が偏る |
| Feature Attribution ドリフト | 各特徴量の寄与度が変化 |

---

## パート 2: データキャプチャ（15分）

### ステップ 2.1: 構成の確認（dry-run）

```bash
python data_capture.py --dry-run
```

### ステップ 2.2: データキャプチャ付きエンドポイントのデプロイ

Model Monitor デモに進むため、エンドポイントを残して実行します：

```bash
python data_capture.py --keep
```

処理の流れ：
1. モデルをトレーニング
2. `DataCaptureConfig`（キャプチャ率 100%）付きでデプロイ
3. テストデータ 50 件で推論し、S3 にリクエスト/レスポンスをキャプチャ
4. エンドポイント名を `endpoint_name.txt` に保存（次のステップで参照）

> ⚠️ `--keep` のためエンドポイントが残ります。パート4のクリーンアップを忘れずに。

### ステップ 2.3: モニタリング手順の全体像

```
1. データキャプチャ開始 → 2. ベースライン作成 →
3. モニタリングジョブ定義/スケジュール → 4. CloudWatch 統合 →
5. 結果の解釈（constraint_violations.json）
```

---

## パート 3: Model Monitor（15分）

### ステップ 3.1: 構成の確認（dry-run）

```bash
python model_monitor.py --dry-run
```

### ステップ 3.2: ベースライン作成とスケジュール設定

```bash
python model_monitor.py
```

処理の流れ：
1. **ベースラインジョブ**: トレーニングデータから `statistics.json` と `constraints.json` を生成
2. **モニタリングスケジュール**: 時間単位のデータ品質検査ジョブを作成（`mle-handson-monitor-schedule`）
3. **CloudWatch 統合**: メトリクス発行を有効化

### ステップ 3.3: 結果の解釈

`constraint_violations.json` に記録される違反チェック：
- `data_type_check` / `completeness_check`
- `baseline_drift_check`（分布のドリフト）
- `missing_column_check` / `extra_column_check`
- `categorical_values_check`

> ⏱️ 最初のモニタリングジョブは次の正時に自動実行されます。
> デモ中は、ベースライン結果（statistics.json / constraints.json）の確認までで十分です。

### ステップ 3.4: 自動修復

CloudWatch アラームをトリガーに：
- ステークホルダー通知（SNS）
- データ分析
- モデル再トレーニング（イベント駆動 / オンデマンド / 予定済み）
- オートスケーリング

---

## パート 4: クリーンアップ（必須）

```bash
cd ~/handson
bash cleanup_all.sh
```

モニタリングスケジュールとエンドポイントが削除されることを確認してください：

```bash
aws sagemaker list-monitoring-schedules --name-contains mle-handson
aws sagemaker list-endpoints --name-contains mle-handson
```

---

## パート 5: まとめ（5分）

- ドリフトの 4 種類（データ品質 / モデル品質 / バイアス / Feature Attribution）
- データキャプチャ → ベースライン → モニタリング → CloudWatch → 自動修復の流れ
- ベースライン制約との比較によるドリフト検出
- 本番の ML ソリューションを継続的に監視する重要性

これでコース全体の ML ライフサイクル（データ処理 → 特徴量 → トレーニング → 評価 → デプロイ → MLOps → モニタリング）のハンズオンが完了です。

---

## 参考ドキュメント

- [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
- [データキャプチャ](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture.html)
- [データ品質モニタリング](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html)
- [CloudWatch との統合](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-interpreting-cloudwatch.html)
