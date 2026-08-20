"""원본 CSV를 DuckDB 웨어하우스로 변환.

csv 파일은 단일한 진실의 원천으로 두고 멱등하게 구성.

플로우
  원본 CSV
    -> transaction 적재 (schema.COLUMN_TYPES)
    -> identity 적재 (하이픈 정규화 후 LEFT JOIN, 커버리지 24%)
    -> dataset_split 부여 / 파생 시간 피처
    -> warehouse.duckdb : txn
"""

import duckdb

from ieee_cis.config import WAREHOUSE_DIR, WAREHOUSE_PATH
from ieee_cis.etl.identity_fix import (
    normalize_identity_columns,
    verify_identity_schema,
)
from ieee_cis.etl.schema import (
    COLUMN_TYPES,
    EXPECTED_ROWS,
    IDENTITY_COLUMN_TYPES,
)
from ieee_cis.etl.time_features import derived_columns_sql


def _read_csv_sql(path: str, types: dict[str, str]) -> str:
    """타입을 명시한 read_csv 호출을 만듦

    자동 추론 -> 메모리 3배 이상 차이
    """
    cols = ", ".join(f"'{c}': '{t}'" for c, t in types.items())
    return f"read_csv('{path}', columns={{{cols}}}, header=true)"


def _identity_select(con: duckdb.DuckDBPyConnection, path: str) -> str:
    """identity CSV 를 정규화된 컬럼명으로 읽는 SELECT 를 만듦

    test_identity.csv 는 id-01 형태의 하이픈을 사용. 정규화 필수.
    """
    raw = [
        r[0]
        for r in con.execute(f"DESCRIBE SELECT * FROM read_csv('{path}')").fetchall()
    ]
    fixed = normalize_identity_columns(raw)
    verify_identity_schema(fixed)

    # 원본 이름으로 읽되 타입은 정규화된 이름 기준으로 매핑.
    types = {old: IDENTITY_COLUMN_TYPES[new] for old, new in zip(raw, fixed)}
    aliases = ", ".join(f'"{old}" AS "{new}"' for old, new in zip(raw, fixed))
    return f"SELECT {aliases} FROM {_read_csv_sql(path, types)}"


def _split_sql(con: duckdb.DuckDBPyConnection, raw_dir, split: str) -> str:
    """한 split의 transaction + identity 조인 SELECT 만듦"""
    # train 은 원본 isFraud 를 빼고 아래에서 다시 넣어 test 와 컬럼 순서를 맞춤
    # test 에는 isFraud 컬럼이 없으므로 EXCLUDE 절 자체를 쓰지 않음.
    exclude = " EXCLUDE (isFraud)" if split == "train" else ""
    txn_path = (raw_dir / f"{split}_transaction.csv").as_posix()
    ident_path = (raw_dir / f"{split}_identity.csv").as_posix()

    # test에는 isFraud가 없음. NULL로 채움.
    # NULL은 "사기 아님"이 아니라 "모름"을 의미.
    txn_types = {
        c: t for c, t in COLUMN_TYPES.items() if split == "train" or c != "isFraud"
    }
    label = "t.isFraud" if split == "train" else "CAST(NULL AS UTINYINT) AS isFraud"

    ident_cols = ", ".join(
        f'i."{c}"' for c in IDENTITY_COLUMN_TYPES if c != "TransactionID"
    )

    return f"""
        SELECT
            t.*{exclude},
            {label},
            '{split}' AS dataset_split,
            (i.TransactionID IS NOT NULL) AS has_identity,
            {ident_cols},
            {derived_columns_sql("t.TransactionDT")}
        FROM {_read_csv_sql(txn_path, txn_types)} t
        LEFT JOIN ({_identity_select(con, ident_path)}) i
                ON t.TransactionID = i.TransactionID
    """


def build(raw_dir, out_path=WAREHOUSE_PATH) -> None:
    """웨어하우스 만듦. 기존 파일이 있으면 txn 테이블을 대체."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out_path))
    try:
        con.execute("SET memory_limit='3GB'")
        con.execute(
            f"CREATE OR REPLACE TABLE txn AS "
            f"{_split_sql(con, raw_dir, 'train')}"
            f"UNION ALL BY NAME "
            f"{_split_sql(con, raw_dir, 'test')}"
        )
        _verify(con)
    finally:
        con.close()


def _verify(con: duckdb.DuckDBPyConnection) -> None:
    """적재 결과 검증. 실패 시 에러."""
    actual = dict(
        con.execute("SELECT dataset_split, COUNT(*) FROM txn GROUP BY 1").fetchall()
    )
    if actual != EXPECTED_ROWS:
        raise ValueError(f"행 수 불일치: 기대 {EXPECTED_ROWS}, 실제 {actual}")

    labeled, unlabeled = con.execute(
        "SELECT COUNT(isFraud), COUNT(*) - COUNT(isFraud) FROM txn"
    ).fetchone()
    if labeled != EXPECTED_ROWS["train"] or unlabeled != EXPECTED_ROWS["test"]:
        raise ValueError(
            f"라벨 분포 이상: labeled={labeled:,}, unlabeled={unlabeled:,}"
        )

    # identity 커버리지는 약 24%. 0% 면 의심.
    (coverage,) = con.execute("SELECT AVG(has_identity::TINYINT) FROM txn").fetchone()
    if not 0.2 < coverage < 0.3:
        raise ValueError(f"identity 커버리지 이상: {coverage:.1%} (기대 24% 내외)")


def main() -> None:
    from ieee_cis.config import RAW_DIR

    print(f"웨어하우스 생성 중... {WAREHOUSE_PATH}")
    build(RAW_DIR)
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    rows = con.execute("SELECT COUNT(*) FROM txn").fetchone()[0]
    cols = len(con.execute("DESCRIBE txn").fetchall())
    size = WAREHOUSE_PATH.stat().st_size / 1e9
    con.close()
    print(f"완료: {rows:,}행 x {cols} 컬럼, {size:.2f}GB")


if __name__ == "__main__":
    main()
