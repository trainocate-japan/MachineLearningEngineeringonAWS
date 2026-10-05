"""
共通ヘルパー: 生成した画像/ファイルを S3 にアップロードし、署名付き URL を発行する

EC2（Session Manager 専用・インバウンド遮断）環境でも、生成した PNG などを
ブラウザから確認できるようにするためのユーティリティです。公開バケットや
ポート開放は不要で、一時的な署名付き URL でのみ共有します。

使い方（CLI）:
    python common/publish.py <ファイルパス> [<ファイルパス> ...]
    python common/publish.py M03-data-processing/output/*.png --expires 3600

使い方（他スクリプトから）:
    from common.publish import publish_file
    url = publish_file("M03-data-processing/output/hist_age.png")

アップロード先: SageMaker デフォルトバケットの mle-handson/published/ 配下
署名付き URL の既定有効期限: 900 秒（15 分、S3 presign の最大値）
"""

import os
import sys
import argparse
import mimetypes


PREFIX = "mle-handson/published"


def _session_and_bucket():
    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    sts = boto3.client("sts", region_name=region)
    account = sts.get_caller_identity()["Account"]
    bucket = f"sagemaker-{region}-{account}"
    s3 = boto3.client("s3", region_name=region)
    # SageMaker デフォルトバケットが無ければ作成
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    return s3, bucket, region


def publish_file(path: str, expires: int = 900) -> str:
    """ファイルを S3 にアップロードし、署名付き URL を返す。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    s3, bucket, region = _session_and_bucket()
    key = f"{PREFIX}/{os.path.basename(path)}"
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    s3.upload_file(path, bucket, key, ExtraArgs={"ContentType": content_type})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=min(expires, 900),  # presign の上限は 900 秒
    )
    return url


def main():
    parser = argparse.ArgumentParser(description="画像/ファイルを S3 にアップしブラウザ閲覧用の署名付き URL を発行")
    parser.add_argument("paths", nargs="+", help="アップロードするファイル（複数可）")
    parser.add_argument("--expires", type=int, default=900, help="有効期限（秒、最大 900）")
    args = parser.parse_args()

    print("=" * 60)
    print(" 署名付き URL の発行（ブラウザ閲覧用）")
    print("=" * 60)

    any_ok = False
    for p in args.paths:
        try:
            url = publish_file(p, args.expires)
            print(f"\n  ファイル: {p}")
            print(f"  URL（{min(args.expires, 900)} 秒有効）:")
            print(f"  {url}")
            any_ok = True
        except FileNotFoundError:
            print(f"\n  ⚠️  見つかりません: {p}")
        except Exception as e:
            print(f"\n  ⚠️  アップロード失敗: {p}\n     {e}")

    if any_ok:
        print("\n  ブラウザで上記 URL を開くと画像を確認できます（期限切れ後は再発行してください）。")
    print()


if __name__ == "__main__":
    main()
