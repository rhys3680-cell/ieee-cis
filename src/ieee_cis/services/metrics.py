"""모니터링 지표.

라벨 의존 지표(사기율 등)는 반드시 dataset_split='train' 으로 제한한다.
test 의 isFraud 는 NULL 이므로 섞으면 전 기간 지표가 조용히 틀어진다.
labeled 건수를 함께 반환하는 것도 그래서다 — 화면이 분모를 알아야
"3.4% (라벨 59만건 기준)"처럼 정직하게 표시할 수 있다.
"""

import duckdb
import pandas as pd

from ieee_cis.ml.train import MODEL_VERSION


def daily(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """일자별 집계 전체.

    395 행뿐이라 한 번 가져와 캐시하면 슬라이더를 움직여도 재쿼리가
    필요 없다.

    Returns:
        day_index, dataset_split, txn_count, total_amount, avg_score,
        alert_count(위험도 0.5 이상), labeled, known_frauds, fraud_rate
    """
    return con.execute(
        """
        SELECT t.day_index,
               ANY_VALUE(t.dataset_split) AS dataset_split,
               COUNT(*) AS txn_count,
               SUM(t.TransactionAmt) AS total_amount,
               AVG(s.score) AS avg_score,
               SUM(CASE WHEN s.score >= 0.5 THEN 1 ELSE 0 END) AS alert_count,
               SUM(CASE WHEN s.score >= 0.9 THEN 1 ELSE 0 END) AS critical_count,
               COUNT(t.isFraud) AS labeled,
               SUM(t.isFraud) AS known_frauds,
               -- 라벨 없는 날은 NULL 이 된다. 0 으로 채우지 않는다.
               AVG(t.isFraud) AS fraud_rate
        FROM txn t JOIN txn_scores s USING (TransactionID)
        WHERE s.model_version = ?
        GROUP BY t.day_index
        ORDER BY t.day_index
        """,
        [MODEL_VERSION],
    ).df()


def hourly(con: duckdb.DuckDBPyConnection, day: int | None = None) -> pd.DataFrame:
    """시간대별 집계.

    hour_slot 은 실제 시각이 아니라 24 시간 주기 안의 상대 슬롯이다.
    슬롯 7 이 새벽 7 시라는 보장은 없다.
    """
    where = "s.model_version = ?"
    params: list = [MODEL_VERSION]
    if day is not None:
        where += " AND t.day_index = ?"
        params.append(day)

    return con.execute(
        f"""
        SELECT t.hour_slot,
               COUNT(*) AS txn_count,
               AVG(s.score) AS avg_score,
               SUM(CASE WHEN s.score >= 0.5 THEN 1 ELSE 0 END) AS alert_count,
               COUNT(t.isFraud) AS labeled,
               AVG(t.isFraud) AS fraud_rate
        FROM txn t JOIN txn_scores s USING (TransactionID)
        WHERE {where}
        GROUP BY t.hour_slot
        ORDER BY t.hour_slot
        """,
        params,
    ).df()


def day_range(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """데이터에 존재하는 일차의 최소/최대."""
    lo, hi = con.execute("SELECT MIN(day_index), MAX(day_index) FROM txn").fetchone()
    return int(lo), int(hi)