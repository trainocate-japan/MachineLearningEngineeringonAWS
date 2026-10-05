# モジュール 9: AWS ML リソースの保護 - ハンズオン手順

```bash
cd ~/handson/M09-securing-resources
```

このモジュールはデータセット生成に依存しません。

---

## パート 1: アクセス制御の整理（10分）

| 項目 | 内容 |
|------|------|
| アクセスコントロール | IAM グループ + ポリシー、サービスロール |
| ネットワーク | VPC、VPC エンドポイント、セキュリティグループ、ネットワーク ACL |
| 暗号化 | AWS KMS（保管時/転送中）、機密データ（患者情報等） |

**ポイント**:
- 暗号化キーの統合管理 → AWS KMS
- チームの権限管理の最善策 → ポリシーを付けた IAM グループ
- サービスには長期認証情報ではなくサービスロールを使用

---

## パート 2: 最小権限ポリシーの実検証（15分）

### ステップ 2.1: 検証対象の確認（dry-run）

```bash
python verify_iam_policy.py --dry-run
```

評価するアクションと期待結果（allowed / denied）の一覧を確認します。

### ステップ 2.2: Policy Simulator で許可/拒否を評価

```bash
python verify_iam_policy.py
```

処理の流れ：
1. ML 向け最小権限ポリシー（テスト用）を定義
2. `simulate-custom-policy` で代表的なアクションを評価
   - 許可すべき操作（S3 Get/Put、SageMaker 学習作成）→ **allowed**
   - 許可すべきでない操作（S3 バケット削除、IAM ユーザー作成、EC2 終了）→ **denied**
3. 期待どおりかを判定表示

### ステップ 2.3: デプロイ済みロールの実効権限を検証

`$SAGEMAKER_ROLE_ARN` が設定されていれば、自動的に実ロールも検証されます。
明示指定する場合：

```bash
python verify_iam_policy.py --role-arn "$SAGEMAKER_ROLE_ARN"
```

`simulate-principal-policy` で、SageMaker 実行ロールが
- SageMaker / S3 を許可し
- 破壊的な IAM 操作（iam:DeleteRole 等）を許可しない

ことを確認します。

---

## パート 3: ネットワーク分離の考え方（5分）

- VPC 内に SageMaker を配置し、インターネット直結を避ける
- VPC エンドポイント経由で S3 / SageMaker API にアクセス
- セキュリティグループ / ネットワーク ACL でトラフィックを制御
- AWS KMS で保管時/転送中のデータを暗号化

> 本ハンズオンのデモ EC2 は Session Manager 専用で、インバウンドを全遮断しています。
> これ自体が「不要なネットワーク露出を避ける」実装例になっています。

---

## まとめ

- 最小権限は「書く」だけでなく「シミュレートして検証する」ことが重要
- 実ロールの実効権限を simulate-principal-policy で確認できる
- ネットワーク分離・暗号化と組み合わせて多層防御を構成する

---

## 参考ドキュメント

- [IAM Policy Simulator](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_testing-policies.html)
- [SageMaker セキュリティ](https://docs.aws.amazon.com/sagemaker/latest/dg/security.html)
- [SageMaker VPC 構成](https://docs.aws.amazon.com/sagemaker/latest/dg/infrastructure-give-access.html)
- [AWS KMS](https://docs.aws.amazon.com/kms/latest/developerguide/overview.html)
