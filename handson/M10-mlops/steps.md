# モジュール 10: MLOps と自動デプロイ - ハンズオン手順

## 前提: データが生成済みであること

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000
cd ~/handson/M10-mlops
```

> このモジュールのパイプラインは生データ `raw_income.csv` を入力とし、
> M04 の `processing_entry.py` を前処理ステップとして再利用します。

---

## パート 1: MLOps の概要（10分）

### DevOps と MLOps の違い

| 機能 | DevOps | MLOps |
|------|--------|-------|
| コードバージョニング | ✓ | ✓ |
| CI/CD | ✓ | ✓ |
| 本番モニタリング | ✓ | ✓ |
| データプロベナンス | | ✓ |
| データセット/モデルのバージョン | | ✓ |
| モデル構築/デプロイワークフロー | | ✓ |

MLOps は「コード」に加えて「データ」と「モデル」のバージョン管理・再現性が重要です。

---

## パート 2: SageMaker Pipelines（25分）

### ステップ 2.1: パイプライン構成の確認（dry-run）

```bash
python sagemaker_pipeline.py --dry-run
```

4 ステップ構成を確認します：
1. **PreprocessData**: 前処理（train/validation/test 分割）
2. **TrainModel**: 組み込み XGBoost で学習
3. **EvaluateModel**: テストデータで AUC を評価
4. **CheckAUC → RegisterModel**: AUC が閾値（0.80）以上なら Model Registry に登録

### ステップ 2.2: パイプラインの作成と実行

```bash
python sagemaker_pipeline.py
```

- `pipeline.upsert()` でパイプライン定義を登録
- `pipeline.start()` で実行し、各ステップのステータスを表示

> ⏱️ 全ステップで 10〜20 分かかります。
> 定義だけ作って Studio で実行を見せたい場合は `--register-only` を使います。

### ステップ 2.3: パイプラインの確認

- SageMaker Studio → Pipelines で DAG を可視化
- 各ステップの入出力アーティファクトを確認
- 条件分岐（AUC 閾値）の動作を確認

---

## パート 3: Model Registry（10分）

### ステップ 3.1: 登録モデルの一覧

```bash
python model_registry.py
```

`mle-handson-model-group` に登録されたモデルパッケージと承認ステータスを確認します。

### ステップ 3.2: モデルの承認

```bash
python model_registry.py --approve
```

最新の `PendingManualApproval` モデルを `Approved` に変更します。

### ステップ 3.3: 承認ワークフローのディスカッション

```
PendingManualApproval → (レビュー) → Approved → 本番デプロイ(CI/CD トリガー)
```

- バージョン管理・ガバナンス・監査可能性
- 承認されたモデルのみ本番にデプロイする統制

---

## パート 4: まとめ（5分）

- MLOps = コード + データ + モデルの再現性・自動化
- SageMaker Pipelines による ML ワークフロー自動化
- Model Registry による承認ワークフロー

---

## クリーンアップ

```bash
cd ~/handson
bash cleanup_all.sh
```

パイプラインと Model Package Group が削除されます。

---

## 参考ドキュメント

- [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [SageMaker Projects](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html)
