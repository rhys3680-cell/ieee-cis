"""시간 피처 파생 테스트."""

import pytest

from ieee_cis.etl.time_features import (
    DERIVED_COLUMNS,
    day_index,
    derived_columns_sql,
    hour_slot,
)

# 실제 데이터 MIN, MAX 값
TRAIN_MIN, TRAIN_MAX = 86_400, 15_811_131
TEST_MIN, TEST_MAX = 18_403_224, 34_214_345


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0, 0),
        (3_599, 0),
        (3_600, 1),
        (23 * 3_600, 23),
        (86_400, 0),
        (86_400 + 7 * 3_600, 7),
    ],
)
def test_hour_slot(offset: int, expected: int) -> None:
    assert hour_slot(offset) == expected


@pytest.mark.parametrize("offset", [0, TRAIN_MIN, TRAIN_MAX, TEST_MIN, TEST_MAX])
def test_hour_slot_always_in_range(offset: int) -> None:
    assert 0 <= hour_slot(offset) <= 23


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0, 0),
        (86_399, 0),
        (86_400, 1),
        (TRAIN_MAX, 182),
    ],
)
def test_day_index(offset: int, expected: int) -> None:
    assert day_index(offset) == expected


def test_train_spans_182_days() -> None:
    "train은 1일차부터 182일차까지."
    assert day_index(TRAIN_MIN) == 1
    assert day_index(TRAIN_MAX) == 182


def test_test_starts_after_train_with_gap() -> None:
    """train 과 test 사이 공백."""
    assert day_index(TEST_MIN) == 213
    assert day_index(TEST_MIN) - day_index(TRAIN_MAX) == 31


def test_sql_defines_all_derived_columns() -> None:
    sql = derived_columns_sql()
    for col in DERIVED_COLUMNS:
        assert f"AS {col}" in sql


def test_sql_matches_python_implementation() -> None:
    """SQL 과 파이썬 구현이 같은 값을 내는지 DuckDB 로 확인한다."""
    duckdb = pytest.importorskip("duckdb")
    offsets = [0, 3_599, 86_400, TRAIN_MIN, TRAIN_MAX, TEST_MAX]

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t (TransactionDT BIGINT)")
    con.executemany("INSERT INTO t VALUES (?)", [(o,) for o in offsets])
    rows = con.execute(
        f"SELECT TransactionDT, {derived_columns_sql()} FROM t ORDER BY 1"
    ).fetchall()

    for offset, slot, day in rows:
        assert slot == hour_slot(offset), f"hour_slot 불일치: offset={offset}"
        assert day == day_index(offset), f"day_index 불일치: offset={offset}"
