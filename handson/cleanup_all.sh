#!/bin/bash
# =============================================================================
# Machine Learning Engineering on AWS - 全リソースクリーンアップ
# =============================================================================
# 全モジュールで作成した AWS リソースを一括削除するスクリプト
# 冪等: リソースが存在しなければスキップ
# =============================================================================

set -e

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")

# ハンズオンで使用するリソースの命名規則（スクリプトと一致させる）
ENDPOINT_PREFIX="mle-handson"
PIPELINE_NAME="mle-handson-pipeline"
MODEL_PACKAGE_GROUP="mle-handson-model-group"
MONITOR_SCHEDULE_PREFIX="mle-handson-monitor"

echo "=============================================="
echo " リソースクリーンアップ"
echo " Machine Learning Engineering on AWS"
echo "=============================================="
echo ""
echo "  アカウント: $ACCOUNT_ID"
echo "  リージョン: $REGION"
echo ""
echo "  対象モジュール: M04〜M11（AWS リソース）"
echo ""
read -p "  続行しますか？ (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "  キャンセルしました。"
    exit 0
fi

echo ""

# =============================================================================
# M11: モニタリング関連リソース
# =============================================================================
echo "──────────────────────────────────────────────"
echo " [M11] モニタリング関連リソースの削除"
echo "──────────────────────────────────────────────"

echo "  モニタリングスケジュールの削除..."
SCHEDULES=$(aws sagemaker list-monitoring-schedules --region "$REGION" \
    --name-contains "$MONITOR_SCHEDULE_PREFIX" \
    --query "MonitoringScheduleSummaries[].MonitoringScheduleName" \
    --output text 2>/dev/null || echo "")
if [ -n "$SCHEDULES" ] && [ "$SCHEDULES" != "None" ]; then
    for s in $SCHEDULES; do
        aws sagemaker delete-monitoring-schedule --monitoring-schedule-name "$s" --region "$REGION" 2>/dev/null && \
            echo "  ✅ モニタリングスケジュール削除: $s" || true
    done
else
    echo "  ─ モニタリングスケジュールなし（スキップ）"
fi

echo ""

# =============================================================================
# M08/M11: SageMaker エンドポイント（最も課金に影響するため優先削除）
# =============================================================================
echo "──────────────────────────────────────────────"
echo " [M08/M11] SageMaker エンドポイントの削除"
echo "──────────────────────────────────────────────"

echo "  リアルタイムエンドポイントの削除..."
ENDPOINTS=$(aws sagemaker list-endpoints --region "$REGION" \
    --name-contains "$ENDPOINT_PREFIX" \
    --query "Endpoints[].EndpointName" --output text 2>/dev/null || echo "")
if [ -n "$ENDPOINTS" ] && [ "$ENDPOINTS" != "None" ]; then
    for ep in $ENDPOINTS; do
        aws sagemaker delete-endpoint --endpoint-name "$ep" --region "$REGION" 2>/dev/null && \
            echo "  ✅ エンドポイント削除: $ep" || true
    done
else
    echo "  ─ エンドポイントなし（スキップ）"
fi

echo "  エンドポイント設定の削除..."
EP_CONFIGS=$(aws sagemaker list-endpoint-configs --region "$REGION" \
    --name-contains "$ENDPOINT_PREFIX" \
    --query "EndpointConfigs[].EndpointConfigName" --output text 2>/dev/null || echo "")
if [ -n "$EP_CONFIGS" ] && [ "$EP_CONFIGS" != "None" ]; then
    for cfg in $EP_CONFIGS; do
        aws sagemaker delete-endpoint-config --endpoint-config-name "$cfg" --region "$REGION" 2>/dev/null && \
            echo "  ✅ エンドポイント設定削除: $cfg" || true
    done
else
    echo "  ─ エンドポイント設定なし（スキップ）"
fi

echo ""

# =============================================================================
# M05/M06/M08: SageMaker モデル
# =============================================================================
echo "──────────────────────────────────────────────"
echo " [M05/M06/M08] SageMaker モデルの削除"
echo "──────────────────────────────────────────────"

MODELS=$(aws sagemaker list-models --region "$REGION" \
    --name-contains "$ENDPOINT_PREFIX" \
    --query "Models[].ModelName" --output text 2>/dev/null || echo "")
if [ -n "$MODELS" ] && [ "$MODELS" != "None" ]; then
    for m in $MODELS; do
        aws sagemaker delete-model --model-name "$m" --region "$REGION" 2>/dev/null && \
            echo "  ✅ モデル削除: $m" || true
    done
else
    echo "  ─ モデルなし（スキップ）"
fi

echo ""

# =============================================================================
# M10: SageMaker Pipeline と Model Package Group
# =============================================================================
echo "──────────────────────────────────────────────"
echo " [M10] Pipeline / Model Registry の削除"
echo "──────────────────────────────────────────────"

echo "  SageMaker Pipeline の削除..."
PIPELINE_EXISTS=$(aws sagemaker describe-pipeline --pipeline-name "$PIPELINE_NAME" --region "$REGION" \
    --query "PipelineName" --output text 2>/dev/null || echo "NONE")
if [ "$PIPELINE_EXISTS" != "NONE" ]; then
    aws sagemaker delete-pipeline --pipeline-name "$PIPELINE_NAME" --region "$REGION" 2>/dev/null && \
        echo "  ✅ Pipeline 削除: $PIPELINE_NAME" || echo "  ⚠️  Pipeline 削除失敗"
else
    echo "  ─ Pipeline なし（スキップ）"
fi

echo "  Model Package Group の削除..."
GROUP_EXISTS=$(aws sagemaker describe-model-package-group \
    --model-package-group-name "$MODEL_PACKAGE_GROUP" --region "$REGION" \
    --query "ModelPackageGroupName" --output text 2>/dev/null || echo "NONE")
if [ "$GROUP_EXISTS" != "NONE" ]; then
    # グループ内のモデルパッケージを先に削除
    PKGS=$(aws sagemaker list-model-packages \
        --model-package-group-name "$MODEL_PACKAGE_GROUP" --region "$REGION" \
        --query "ModelPackageSummaryList[].ModelPackageArn" --output text 2>/dev/null || echo "")
    for pkg in $PKGS; do
        aws sagemaker delete-model-package --model-package-name "$pkg" --region "$REGION" 2>/dev/null && \
            echo "  ✅ Model Package 削除: $pkg" || true
    done
    sleep 3
    aws sagemaker delete-model-package-group \
        --model-package-group-name "$MODEL_PACKAGE_GROUP" --region "$REGION" 2>/dev/null && \
        echo "  ✅ Model Package Group 削除: $MODEL_PACKAGE_GROUP" || echo "  ⚠️  削除失敗"
else
    echo "  ─ Model Package Group なし（スキップ）"
fi

echo ""

# =============================================================================
# S3: ハンズオン出力（SageMaker デフォルトバケットのプレフィックス）
# =============================================================================
echo "──────────────────────────────────────────────"
echo " [共通] S3 ハンズオン出力の削除"
echo "──────────────────────────────────────────────"

SM_BUCKET="sagemaker-${REGION}-${ACCOUNT_ID}"
if aws s3 ls "s3://$SM_BUCKET" 2>/dev/null >/dev/null; then
    for prefix in "mle-handson" "mle-processing" "mle-training" "mle-tuning" "mle-monitor"; do
        if aws s3 ls "s3://$SM_BUCKET/$prefix/" 2>/dev/null >/dev/null; then
            aws s3 rm "s3://$SM_BUCKET/$prefix/" --recursive 2>/dev/null && \
                echo "  ✅ S3 出力削除: s3://$SM_BUCKET/$prefix/" || true
        fi
    done
else
    echo "  ─ SageMaker デフォルトバケットなし（スキップ）"
fi

echo ""

# =============================================================================
# ローカル生成物（EC2 内）
# =============================================================================
echo "──────────────────────────────────────────────"
echo " [共通] ローカル生成物の削除"
echo "──────────────────────────────────────────────"

HANDSON_ROOT="$(cd "$(dirname "$0")" && pwd)"
find "$HANDSON_ROOT" -type d -name "data" -prune -exec echo "  ─ 残置（手動削除可）: {}" \; 2>/dev/null || true
echo "  ℹ️  ローカルの data/ output/ は課金に影響しないため残置します。"
echo "     完全に消す場合: find $HANDSON_ROOT -type d \( -name data -o -name output \) -exec rm -rf {} +"

echo ""

# =============================================================================
# 完了
# =============================================================================
echo "=============================================="
echo " クリーンアップ完了!"
echo "=============================================="
echo ""
echo "  ⚠️  以下は手動確認を推奨します:"
echo "  - SageMaker トレーニング / チューニング / 処理ジョブ（実行中の場合のみ課金）"
echo "  - CloudWatch ロググループ (/aws/sagemaker/*)"
echo ""
echo "  実行中ジョブの確認:"
echo "    aws sagemaker list-training-jobs --status-equals InProgress --region $REGION"
echo "    aws sagemaker list-processing-jobs --status-equals InProgress --region $REGION"
echo ""
