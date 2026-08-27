"""알람큐 조회.

위험도 순으로 정렬된 거래 목록을 페이지 단위로 제공한다.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd

from ieee_cis.ml.train import MODEL_VERSION

#: 알람큐 목록에 표시할 컬럼과 한국어 라벨.
#: MVP 범위. 나중에 semantic/columns.yaml 로 옮긴다.
QUEUE_COLUMNS = {
    "TransactionID": "거래번호",
    "score": "위험도",
    "TransactionAmt": "금액",
    "ProductCD": "상품코드",
    "card1": "카드ID",
    "card4": "카드사",
    "card6": "카드종류",
    "addr1": "청구지역",
    "P_emaildomain": "구매자이메일",
    "DeviceType": "기기유형",
    "hour_slot": "시간대",
    "day_index": "일차",
    "dataset_split": "구분",
    "isFraud": "실제사기",
}


@dataclass
class AlertFilters:
    """알람큐 필터. 모두 선택 사항이다."""

    min_score: float = 0.0
    max_score: float = 1.0
    product_cd: tuple[str, ...] = ()
    split: str | None = None  # 'train' / 'test' / None(전체)
    min_amount: float | None = None
    #: 특정 일차만. 실제 FDS 운영은 "그날 들어온 알람"을 처리하므로
    #: 395 일치를 한 화면에 섞어 보여주면 업무 구조와 맞지 않는다.
    day: int | None = None

    def where(self) -> tuple[str, list]:
        """WHERE 절과 파라미터를 만든다.

        값을 SQL 문자열에 직접 넣지 않고 파라미터로 넘긴다.
        """
        clauses = ["s.model_version = ?"]
        params: list = [MODEL_VERSION]

        clauses.append("s.score BETWEEN ? AND ?")
        params += [self.min_score, self.max_score]

        if self.product_cd:
            placeholders = ", ".join("?" * len(self.product_cd))
            clauses.append(f"t.ProductCD IN ({placeholders})")
            params += list(self.product_cd)
        if self.split:
            clauses.append("t.dataset_split = ?")
            params.append(self.split)
        if self.min_amount is not None:
            clauses.append("t.TransactionAmt >= ?")
            params.append(self.min_amount)
        if self.day is not None:
            clauses.append("t.day_index = ?")
            params.append(self.day)

        return " AND ".join(clauses), params


def fetch_page(
    con: duckdb.DuckDBPyConnection,
    filters: AlertFilters | None = None,
    offset: int = 0,
    limit: int = 100,
) -> pd.DataFrame:
    """위험도 순 알람큐 한 페이지."""
    filters = filters or AlertFilters()
    where, params = filters.where()
    cols = ", ".join(f"s.{c}" if c == "score" else f"t.{c}" for c in QUEUE_COLUMNS)
    return con.execute(
        f"""
        SELECT {cols}
        FROM txn t JOIN txn_scores s USING (TransactionID)
        WHERE {where}
        ORDER BY s.score DESC, t.TransactionID
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).df()


def queue_stats(
    con: duckdb.DuckDBPyConnection, filters: AlertFilters | None = None
) -> dict:
    """큐 요약. 총 건수와 위험등급별 분포.

    라벨 의존 지표는 train 행만으로 계산한다. test 의 isFraud 는 NULL
    이므로 섞으면 사기율이 조용히 틀어진다.
    """
    filters = filters or AlertFilters()
    where, params = filters.where()
    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN s.score >= 0.9 THEN 1 ELSE 0 END) AS critical,
            SUM(CASE WHEN s.score >= 0.5 AND s.score < 0.9 THEN 1 ELSE 0 END) AS high,
            SUM(CASE WHEN s.score >= 0.1 AND s.score < 0.5 THEN 1 ELSE 0 END) AS medium,
            SUM(CASE WHEN s.score < 0.1 THEN 1 ELSE 0 END) AS low,
            SUM(t.TransactionAmt) AS total_amount,
            COUNT(t.isFraud) AS labeled,
            SUM(t.isFraud) AS known_frauds
        FROM txn t JOIN txn_scores s USING (TransactionID)
        WHERE {where}
        """,
        params,
    ).fetchone()

    keys = (
        "total",
        "critical",
        "high",
        "medium",
        "low",
        "total_amount",
        "labeled",
        "known_frauds",
    )
    stats = dict(zip(keys, row))

    # 조건에 맞는 행이 없으면 SUM 이 NULL 을 반환한다. COUNT 는 0 이라
    # total 만 정상이고 나머지가 None 이 되어 화면에서 포맷 오류가 난다.
    # 건수는 0 으로 채우되, 사기 관련 값은 "모름"과 "없음"이 다르므로
    # known_frauds 는 라벨이 있을 때만 0 으로 본다.
    for key in ("critical", "high", "medium", "low"):
        stats[key] = int(stats[key] or 0)
    stats["total_amount"] = float(stats["total_amount"] or 0.0)
    stats["labeled"] = int(stats["labeled"] or 0)
    stats["known_frauds"] = (
        int(stats["known_frauds"] or 0) if stats["labeled"] else None
    )

    stats["fraud_rate"] = (
        stats["known_frauds"] / stats["labeled"] if stats["labeled"] else None
    )
    return stats
