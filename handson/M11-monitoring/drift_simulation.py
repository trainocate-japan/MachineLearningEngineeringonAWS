"""
モジュール 11: ドリフトのシミュレーションと検出（ローカル）

本番データの分布を意図的にシフトさせ、ベースライン（トレーニングデータ）からの
統計的な逸脱を検出するロジックをローカルで体験します。これは SageMaker Model
Monitor が内部で行う「ベースライン制約との比較」を簡略化したデモです。

扱う内容:
- ML モデルのドリフト検出（データ品質/モデル品質/バイアス/Feature Attribution）
- データ品質ドリフトのモニタリング手順

AWS API は呼び出しません。
入力: ../common/data/features_income.csv
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "common", "data")
IN_CSV = os.path.join(DATA_DIR, "features_income.csv")

# ドリフト判定の閾値（平均の変化率）
DRIFT_THRESHOLD = 0.10  # 10% を超える平均変化をドリフトとみなす


def build_baseline(df: pd.DataFrame, cols) -> dict:
    """ベースライン統計（平均・標準偏差）を計算（statistics.json 相当）"""
    return {c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in cols}


def simulate_production(df: pd.DataFrame) -> pd.DataFrame:
    """本番データのドリフトをシミュレート:
    - age_std を全体的に上方シフト（高齢化 = データ品質ドリフト）
    - hours_std のばらつきを拡大
    - 新しい州カテゴリの出現を模して state 列の一部を 0 埋め変更
    """
    prod = df.copy()
    rng = np.random.default_rng(7)
    # age_std: 平均を上方シフト（高齢化）
    prod["age_std"] = prod["age_std"] + 0.8
    # hours_std: 平均を下方シフト + ばらつき拡大（労働時間の減少とばらつき増）
    prod["hours_std"] = prod["hours_std"] * 1.5 - 0.5 + rng.normal(0, 0.2, len(prod))
    return prod


def detect_drift(baseline: dict, prod: pd.DataFrame) -> list:
    """ベースライン平均と本番平均を比較し、閾値超過をドリフトとして報告"""
    results = []
    for col, stats in baseline.items():
        base_mean = stats["mean"]
        prod_mean = float(prod[col].mean())
        denom = abs(base_mean) if abs(base_mean) > 1e-9 else 1.0
        change = abs(prod_mean - base_mean) / denom
        drifted = change > DRIFT_THRESHOLD
        results.append((col, base_mean, prod_mean, change, drifted))
    return results


def main():
    print("=" * 60)
    print(" ドリフトのシミュレーションと検出（ローカルデモ）")
    print("=" * 60)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"データが見つかりません: {IN_CSV}\n"
            "先に M04 の feature_transform.py を実行してください。"
        )

    df = pd.read_csv(IN_CSV)
    monitored_cols = ["age_std", "hours_std", "education_num_norm", "income_log"]

    # ベースライン作成（トレーニングデータの統計）
    baseline = build_baseline(df, monitored_cols)
    print("\n  [ステップ 2 相当] ベースライン統計を作成（statistics.json 相当）")
    for col, s in baseline.items():
        print(f"    {col:20s}: mean={s['mean']:+.3f}, std={s['std']:.3f}")

    # 本番データのドリフトをシミュレート
    prod = simulate_production(df)

    # ドリフト検出（ベースライン制約との比較）
    print("\n  [ステップ 5 相当] ベースラインと本番データを比較しドリフト検出")
    print(f"  判定閾値: 平均変化率 > {DRIFT_THRESHOLD:.0%}")
    print("\n  特徴量               base_mean  prod_mean   変化率   判定")
    print("  " + "─" * 58)
    violations = 0
    for col, bm, pm, change, drifted in detect_drift(baseline, prod):
        mark = "⚠️ DRIFT" if drifted else "OK"
        if drifted:
            violations += 1
        print(f"  {col:20s} {bm:+8.3f}  {pm:+8.3f}  {change:6.1%}   {mark}")

    print("\n  [検出結果]")
    if violations:
        print(f"  {violations} 個の特徴量でベースラインドリフトを検出 "
              "(baseline_drift_check 違反に相当)")
        print("  → 本番では CloudWatch アラーム → SNS 通知 → 再トレーニングをトリガー")
    else:
        print("  ドリフトは検出されませんでした")

    print("\n  [ドリフトの種類]")
    print("  - データ品質ドリフト:    入力データの統計/スキーマが変化（本デモ）")
    print("  - モデル品質ドリフト:    予測精度が時間とともに劣化")
    print("  - バイアスドリフト:      特定グループへの予測が偏る")
    print("  - Feature Attribution:  各特徴量の寄与度が変化")

    print("\n  [自動修復]")
    print("  - ステークホルダー通知 / データ分析 / モデル再トレーニング / オートスケーリング")
    print("  - 再トレーニング戦略: イベント駆動 / オンデマンド / 予定済み")

    print("\n  ✅ ドリフトシミュレーション完了")


if __name__ == "__main__":
    main()
