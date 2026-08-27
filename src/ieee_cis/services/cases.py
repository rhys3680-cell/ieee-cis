"""건별 심사 조회."""

import duckdb
import pandas as pd

from ieee_cis.ml.train import MODEL_VERSION

#: 케이스 상세에 표시할 컬럼 그룹.
#: 439 개 전부를 보여주면 분석가가 읽을 수 없다. 운영상 의미 있는
#: 것만 고른다. MVP 범위이며 나중에 semantic/columns.yaml 로 옮긴다.
DETAIL_GROUPS = {
    "거래": {
        "TransactionAmt": "금액",
        "amt_decimals": "금액 소수 자릿수",
        "ProductCD": "상품코드",
        "hour_slot": "시간대",
        "day_index": "일차",
    },
    "카드": {
        "card1": "카드ID",
        "card2": "카드코드2",
        "card3": "카드코드3",
        "card4": "카드사",
        "card5": "카드코드5",
        "card6": "카드종류",
    },
    "주소·연락처": {
        "addr1": "청구지역",
        "addr2": "청구국가",
        "dist1": "거리1",
        "dist2": "거리2",
        "P_emaildomain": "구매자 이메일",
        "R_emaildomain": "수취인 이메일",
    },
    "기기": {
        "DeviceType": "기기유형",
        "DeviceInfo": "기기정보",
        "id_30": "운영체제",
        "id_31": "브라우저",
        "id_33": "화면해상도",
        "has_identity": "기기정보 보유",
    },
}

#: 파생 피처는 원본 컬럼이 아니므로 조회 시 별도로 계산한다.
_DERIVED = {"amt_decimals"}


def get_transaction(con: duckdb.DuckDBPyConnection, txn_id: int) -> dict | None:
    """거래 상세. 없으면 None."""
    cols = [c for g in DETAIL_GROUPS.values() for c in g if c not in _DERIVED]
    row = con.execute(
        f"""
        SELECT t.TransactionID, t.dataset_split, t.isFraud,
               s.score, s.model_version,
               CASE
                 WHEN t.TransactionAmt = FLOOR(t.TransactionAmt) THEN 0
                 WHEN ROUND(t.TransactionAmt, 2) = t.TransactionAmt THEN 2
                 ELSE 3
               END AS amt_decimals,
               {", ".join(f't."{c}"' for c in cols)}
        FROM txn t JOIN txn_scores s USING (TransactionID)
        WHERE t.TransactionID = ? AND s.model_version = ?
        """,
        [txn_id, MODEL_VERSION],
    ).df()
    return None if row.empty else row.iloc[0].to_dict()


def get_related(
    con: duckdb.DuckDBPyConnection, txn_id: int, limit: int = 20
) -> pd.DataFrame:
    """같은 카드의 다른 거래.

    분석가가 "이 카드가 평소 어떻게 쓰였나"를 보는 화면이다.
    """
    return con.execute(
        """
        WITH target AS (SELECT card1 FROM txn WHERE TransactionID = ?)
        SELECT t.TransactionID, s.score, t.TransactionAmt, t.ProductCD,
               t.addr1, t.hour_slot, t.day_index, t.dataset_split, t.isFraud
        FROM txn t
        JOIN txn_scores s USING (TransactionID)
        JOIN target ON t.card1 = target.card1
        WHERE s.model_version = ? AND t.TransactionID != ?
        ORDER BY t.day_index DESC, t.TransactionID
        LIMIT ?
        """,
        [txn_id, MODEL_VERSION, txn_id, limit],
    ).df()


def card_summary(con: duckdb.DuckDBPyConnection, txn_id: int) -> dict | None:
    """해당 카드의 이력 요약. 사기율은 라벨 있는 행만으로 계산한다."""
    row = con.execute(
        """
        WITH target AS (SELECT card1 FROM txn WHERE TransactionID = ?)
        SELECT COUNT(*) AS txn_count,
               SUM(t.TransactionAmt) AS total_amount,
               AVG(t.TransactionAmt) AS avg_amount,
               COUNT(t.isFraud) AS labeled,
               SUM(t.isFraud) AS known_frauds,
               MIN(t.day_index) AS first_day,
               MAX(t.day_index) AS last_day
        FROM txn t JOIN target ON t.card1 = target.card1
        """,
        [txn_id],
    ).fetchone()

    if not row or not row[0]:
        return None
    keys = (
        "txn_count",
        "total_amount",
        "avg_amount",
        "labeled",
        "known_frauds",
        "first_day",
        "last_day",
    )
    summary = dict(zip(keys, row))

    # 라벨이 하나도 없는 카드(test 구간에만 등장)는 SUM 이 NULL 을 낸다.
    # "사기 0 건"과 "사기 여부 모름"은 다르므로 labeled 가 있을 때만
    # 0 으로 채우고, 없으면 None 을 유지한다.
    summary["labeled"] = int(summary["labeled"] or 0)
    summary["known_frauds"] = (
        int(summary["known_frauds"] or 0) if summary["labeled"] else None
    )
    summary["fraud_rate"] = (
        summary["known_frauds"] / summary["labeled"] if summary["labeled"] else None
    )
    return summary
