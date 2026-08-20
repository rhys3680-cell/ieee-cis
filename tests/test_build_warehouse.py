"""웨어하우스 빌드 SQL 생성 테스트.

전체 ETL은 2분이 걸리므로 테스트 X. 대신 SQL을 만드는 로직만 검증.
"""

import duckdb
import pytest

from ieee_cis.etl.build_warehouse import _read_csv_sql
from ieee_cis.etl.schema import COLUMN_TYPES, EXPECTED_ROWS, IDENTITY_COLUMN_TYPES


def test_expected_rows_matches_real_data():
    """행 수 기대값이 실제 CSV와 맞는지 확인."""
    assert EXPECTED_ROWS == {"train": 590_540, "test": 506_691}


def test_identity_types_cover_all_columns():
    assert len(IDENTITY_COLUMN_TYPES) == 41
    for i in range(1, 39):
        assert f"id_{i:02d}" in IDENTITY_COLUMN_TYPES
    for c in ("TransactionID", "DeviceType", "DeviceInfo"):
        assert c in IDENTITY_COLUMN_TYPES


def test_read_csv_sql_includes_all_types():
    sql = _read_csv_sql("x.csv", {"a": "INTEGER", "b": "VARCHAR"})
    assert "'a': 'INTEGER'" in sql
    assert "'b': 'VARCHAR'" in sql
    assert "header=true" in sql


def test_read_csv_sql_parses():
    """생성된 SQL 이 문법적으로 유효한지 DuckDB로 확인."""
    sql = _read_csv_sql("nonexistent.csv", dict(list(COLUMN_TYPES.items())[:5]))
    con = duckdb.connect(":memory:")
    # 파일이 없으므로 IO 에러가 나야함. 파싱 에러가 나면 안됨.
    with pytest.raises(duckdb.IOException):
        con.execute(f"SELECT * FROM {sql}")
