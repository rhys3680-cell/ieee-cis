"""학습된 모델로 전체 거래에 점수를 매겨 웨어하우스에 적재.

과거 데이셋이라 실시간 스트림이 없다. 110만 행 전체를 오프라인으로 점수화해
txn_scores 테이블에 넣고 화면은 조회만.

점수는 model_version과 함께 기록
"""

import duckdb
import lightgbm as lgb
import pandas as pd

from ieee_cis.config import MODEL_DIR, WAREHOUSE_PATH
from ieee_cis.ml import features
from ieee_cis.ml.train import MODEL_VERSION


def load_model(version: str = MODEL_VERSION) -> lgb.Booster:
    path = MODEL_DIR / f"lgbm_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"모델이 없습니다: {path}\n먼저 학습하세요: uv run python -m ieee_cis.ml.train"
        )
    return lgb.Booster(model_file=str(path))


def _predict(booster: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    """한 split의 점수를 계산."""
    drop = ["isFraud", "TransactionID", features.SPLIT_COLUMN]
    x = df.drop(columns=[c for c in drop if c in df.columns])

    expected = booster.feature_name()
    missing = set(expected) - set(x.columns)
    if missing:
        raise ValueError(f"학습에 쓴 피처가 없습니다: {sorted(missing)}")
    x = x[expected]

    return pd.DataFrame(
        {"TransactionID": df["TransactionID"].to_numpy(), "score": booster.predict(x)}
    )


def score_all(
    con: duckdb.DuckDBPyConnection,
    booster: lgb.Booster,
    version: str = MODEL_VERSION,
) -> int:
    """train / test 전체를 점수화해 txn_scores 에 적재한다.

    같은 version 의 기존 행은 지우고 다시 넣는다(멱등).

    Returns:
        적재한 행 수.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS txn_scores (
            TransactionID INTEGER NOT NULL,
            score DOUBLE NOT NULL,
            model_version VARCHAR NOT NULL
        )
    """)
    con.execute("DELETE FROM txn_scores WHERE model_version = ?", [version])

    total = 0
    for split in ("train", "test"):
        df = features.load(con, split)
        scored = _predict(booster, df)
        scored["model_version"] = version
        con.execute("INSERT INTO txn_scores SELECT * FROM scored")
        total += len(scored)
        print(f"  {split:5s} {len(scored):>9,} 행")

    return total


def verify(con: duckdb.DuckDBPyConnection, version: str = MODEL_VERSION) -> None:
    """적재 결과를 검증한다."""
    n, lo, hi = con.execute(
        "SELECT COUNT(*), MIN(score), MAX(score) FROM txn_scores WHERE model_version = ?",
        [version],
    ).fetchone()

    expected = con.execute("SELECT COUNT(*) FROM txn").fetchone()[0]
    if n != expected:
        raise ValueError(f"행 수 불일치: 기대 {expected:,}, 실제 {n:,}")
    if not (0 <= lo <= hi <= 1):
        raise ValueError(f"점수 범위 이상: {lo} ~ {hi}")

    # 모든 거래에 점수가 하나씩만 있어야 한다
    dup = con.execute(
        "SELECT COUNT(*) FROM (SELECT TransactionID FROM txn_scores "
        "WHERE model_version = ? GROUP BY 1 HAVING COUNT(*) > 1)",
        [version],
    ).fetchone()[0]
    if dup:
        raise ValueError(f"중복 점수 {dup:,} 건")


def main() -> None:
    booster = load_model()
    con = duckdb.connect(str(WAREHOUSE_PATH))
    try:
        con.execute("SET memory_limit='3GB'")
        print(f"점수 계산 중... (모델 {MODEL_VERSION})")
        total = score_all(con, booster)
        verify(con)

        print(f"\n완료: {total:,} 행")
        print(
            con.execute(f"""
            SELECT t.dataset_split,
                   COUNT(*) AS n,
                   ROUND(AVG(s.score), 4) AS avg_score,
                   ROUND(QUANTILE_CONT(s.score, 0.99), 4) AS p99
            FROM txn t JOIN txn_scores s USING (TransactionID)
            WHERE s.model_version = '{MODEL_VERSION}'
            GROUP BY 1 ORDER BY 1
        """)
            .df()
            .to_string(index=False)
        )
    finally:
        con.close()


if __name__ == "__main__":
    main()
