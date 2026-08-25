"""LightGBM 베이스라인

시간 기반으로 나눈 학습/검증셋을 활용하여 모델 만들기.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ieee_cis.config import MODEL_DIR, WAREHOUSE_PATH
from ieee_cis.ml import features

#: 모델 버전
#: v1 -> v2: is_null_* 16 개와 day_index 제거 (importance_v1.json 근거)
MODEL_VERSION = "v2"

#: 알람 비율
ALERT_BUDGET = (0.001, 0.005, 0.01, 0.05)

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_12": 1.0,
    "verbose": -1,
    "num_threads": 4,
}

NUM_ROUNDS = 2000
EARLY_STOPPING = 100


def _split_xy(df):
    """학습에 쓸 X와 y."""
    drop = ["isFraud", "TransactionID", features.SPLIT_COLUMN]
    x = df.drop(columns=[c for c in drop if c in df.columns])
    y = df["isFraud"].astype("uint8")
    return x, y


def recall_at_k(y_true, y_score, k: float) -> float:
    """상위 k 비율을 심사했을 때 잡히는 사기 비율.

    운영에서 가장 중요한 지표다. 분석가가 볼 수 있는 알람 수는
    정해져 있으므로, 전체 순위가 아니라 상위 구간의 성능이 중요하다.
    """
    n = max(1, int(len(y_score) * k))
    top = np.argsort(y_score)[::-1][:n]
    caught = y_true.iloc[top].sum() if hasattr(y_true, "iloc") else y_true[top].sum()
    total = y_true.sum()
    return float(caught / total) if total else 0.0


def evaluate(y_true, y_score) -> dict:
    """운영 관점 지표를 계산한다."""
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        # 불균형 데이터(사기 3.5%)에서는 ROC-AUC 보다 PR-AUC 가 정직하다.
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }
    for k in ALERT_BUDGET:
        metrics[f"recall@{k:.1%}"] = recall_at_k(y_true, y_score, k)
    return metrics


def train(con: duckdb.DuckDBPyConnection) -> tuple[lgb.Booster, dict]:
    """학습하고 (모델, 지표) 를 반환한다."""
    df = features.load(con, "train")
    train_df, valid_df = features.time_split(df)

    x_tr, y_tr = _split_xy(train_df)
    x_va, y_va = _split_xy(valid_df)

    booster = lgb.train(
        PARAMS,
        lgb.Dataset(x_tr, y_tr),
        num_boost_round=NUM_ROUNDS,
        valid_sets=[lgb.Dataset(x_va, y_va)],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    metrics = {
        "model_version": MODEL_VERSION,
        "best_iteration": booster.best_iteration,
        "n_features": x_tr.shape[1],
        "split_day": features.SPLIT_DAY,
        "train": evaluate(y_tr, booster.predict(x_tr)),
        "valid": evaluate(y_va, booster.predict(x_va)),
    }
    return booster, metrics


def save(booster: lgb.Booster, metrics: dict, model_dir: Path = MODEL_DIR) -> None:
    """모델과 지표를 저장한다."""
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / f"lgbm_{MODEL_VERSION}.txt"))

    (model_dir / f"metrics_{MODEL_VERSION}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 전역 feature importance. 화면 4 와 피처 축소 판단에 쓴다.
    importance = sorted(
        zip(booster.feature_name(), booster.feature_importance("gain")),
        key=lambda x: -x[1],
    )
    (model_dir / f"importance_{MODEL_VERSION}.json").write_text(
        json.dumps(
            [{"feature": f, "gain": float(g)} for f, g in importance],
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    con.execute("SET memory_limit='3GB'")
    try:
        booster, metrics = train(con)
    finally:
        con.close()

    save(booster, metrics)

    v = metrics["valid"]
    print(
        f"\n=== {MODEL_VERSION} (검증 {v['n']:,} 행, 사기율 {v['base_rate']:.3%}) ==="
    )
    print(f"ROC-AUC  {v['roc_auc']:.4f}")
    print(f"PR-AUC   {v['pr_auc']:.4f}")
    for k in ALERT_BUDGET:
        key = f"recall@{k:.1%}"
        print(f"{key:14s} {v[key]:.3f}")
    print(f"\n최적 반복 {metrics['best_iteration']} / 피처 {metrics['n_features']}")

    gap = metrics["train"]["roc_auc"] - v["roc_auc"]
    if gap > 0.15:
        print(f"\n경고: 학습-검증 AUC 격차 {gap:.3f}. 과적합 가능성.")


if __name__ == "__main__":
    main()
