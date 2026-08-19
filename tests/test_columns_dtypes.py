"""컬럼 dtype 정의 테스트."""

import pytest

from ieee_cis.etl.schema import COLUMN_TYPES

VALID_TYPES = {
    "UTINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "DOUBLE",
    "VARCHAR",
    "BOOLEAN",
}


def test_covers_all_transaction_columns():
    assert len(COLUMN_TYPES) == 394


def test_all_types_are_valid():
    unknown = {c: t for c, t in COLUMN_TYPES.items() if t not in VALID_TYPES}
    assert not unknown


def test_key_columns_have_expected_types():
    """자주 쓰는 컬럼은 타입이 바뀌면 곧바로 문제가 되므로 고정."""
    assert COLUMN_TYPES["TransactionID"] == "INTEGER"
    assert COLUMN_TYPES["TransactionDT"] == "INTEGER"
    assert COLUMN_TYPES["isFraud"] == "UTINYINT"
    assert COLUMN_TYPES["TransactionAmt"] == "DOUBLE"


def test_m4_is_varchar_not_boolean():
    """M1-9 중 M4 만 M0/M1/M2 값을 가짐. 일괄 처리 X"""
    assert COLUMN_TYPES["M4"] == "VARCHAR"
    for i in (1, 2, 3, 5, 6, 7, 8, 9):
        assert COLUMN_TYPES[f"M{i}"] == "BOOLEAN"


def test_v_columns_complete():
    for i in range(1, 340):
        assert f"V{i}" in COLUMN_TYPES


def test_matches_real_csv_header():
    """실제 CSV 헤더와 대조"""
    import csv
    from ieee_cis.config import RAW_DIR

    path = RAW_DIR / "train_transaction.csv"
    if not path.exists():
        pytest.skip("원본 데이터 없음")

    with open(path, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert set(header) == set(COLUMN_TYPES)
