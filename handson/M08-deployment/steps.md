# モジュール 8: モデルデプロイ戦略 - ハンズオン手順

## 前提: 特徴量データが生成済みであること

```bash
cd ~/handson
python common/generate_dataset.py --rows 4000
python M04-feature-engineering/data_cleaning.py
python M04-feature-engineering/feature_transform.py
cd ~/handson/M08-deployment
```

> ⚠️ **コスト注意**: このモジュールはリアルタイムエンドポイントを作成します。
> 各デモの最後、または研修終了時に `cleanup_all.sh` を必ず実行してください。

---

## パート 1: デプロイターゲットと推論オプションの選択（10分）

### ステップ 1.1: ディスカッション

**デプロイターゲット**:

| ターゲット | 特徴 |
|-----------|------|
| SageMaker エンドポイント | フルマネージド・運用負荷最小（本ハンズオン） |
| Amazon EKS | カスタム設定・Kubernetes |
| Amazon ECS | マネージドコンテナ・K8s 不要 |
| AWS Lambda | 軽量・サーバーレス・従量課金 |

**推論オプション**:

| オプション | ユースケース |
|-----------|------------|
| リアルタイム | 低レイテンシ・高スループット（レコメンデーション等） |
| サーバーレス | 断続的トラフィック |
| 非同期 | 大きなペイロード・長い処理時間 |
| バッチ変換 | 大量データのオフライン推論 |

---

## パート 2: リアルタイムエンドポイント（20分）

### ステップ 2.1: 構成の確認（dry-run）

```bash
python realtime_endpoint.py --dry-run
```

### ステップ 2.2: デプロイと推論

```bash
python realtime_endpoint.py
```

処理の流れ：
1. 組み込み XGBoost でモデルをトレーニング
2. リアルタイムエンドポイントにデプロイ（`mle-handson-rt-*`）
3. テストデータの先頭 5 件で推論（確率と予測ラベル）
4. **エンドポイントを自動削除**（課金停止）

> 💡 デモで更新作業（パート4）に使うためエンドポイントを残したい場合は `--keep` を付けます。
> ただし課金が続くため、必ず後で削除してください。

### ステップ 2.3: 考察

- エンドポイントのデプロイには数分かかる（モデルのロードとヘルスチェック）
- 推論は低レイテンシ（ミリ秒〜数十ミリ秒）
- 常時起動のため、不要なら削除する

---

## パート 3: バッチ変換（15分）

### ステップ 3.1: 構成の確認（dry-run）

```bash
python batch_transform.py --dry-run
```

### ステップ 3.2: バッチ推論の実行

```bash
python batch_transform.py
```

処理の流れ：
1. モデルをトレーニング
2. Transformer で S3 のテストデータに対しバッチ推論
3. 出力（`.out`）を S3 から取得し先頭 5 件を確認

### ステップ 3.3: リアルタイム vs バッチの比較

| 観点 | リアルタイム | バッチ変換 |
|------|------------|-----------|
| 起動 | 常時 | ジョブ時のみ |
| レイテンシ | 低 | 高（まとめて処理） |
| コスト | 起動中ずっと課金 | ジョブ完了で停止 |
| 削除忘れリスク | あり | なし |

→ バッチはジョブ完了でリソースが解放されるため、削除忘れによる課金が発生しません。

---

## パート 4: ブルー/グリーンとトラフィックシフト（15分）

### ステップ 4.1: トラフィックシフトモードの確認

```bash
python traffic_shifting.py
```

Linear トラフィックシフトのデプロイガードレール構成（JSON）とモード比較表を確認します。

### ステップ 4.2: モードの使い分け

| モード | ブラスト半径 | リスク | 更新時間 |
|-------|------------|-------|---------|
| All at once | 大（100%） | 高 | 最短 |
| Canary | 小（少量先行） | 中 | 中 |
| Linear | 段階的 | 低 | 長 |

全モードで、CloudWatch アラーム連動の**自動ロールバック**が可能です。

### ステップ 4.3（任意）: 実際の更新デモ

```bash
python traffic_shifting.py --run
```

初期エンドポイント（Blue）を作成し、Linear シフトの構成を提示します（コスト抑制のため更新自体はスキップ）。完了後にエンドポイントは自動削除されます。

---

## パート 5: クリーンアップ（必須）

```bash
cd ~/handson
bash cleanup_all.sh
```

**特にリアルタイムエンドポイントが残っていないか確認してください:**

```bash
aws sagemaker list-endpoints --name-contains mle-handson
```

---

## 参考ドキュメント

- [SageMaker 推論オプション](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [バッチ変換](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- [デプロイガードレール（ブルー/グリーン）](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html)
- [トラフィックシフトモード](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green-traffic-shifting-modes.html)
