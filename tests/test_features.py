"""피처 생성 테스트."""

import duckdb
import pandas as pd
import pytest

from ieee_cis.ml.features import (
    CATEGORICAL_COLUMNS,
    NULL_FLAG_COLUMNS,
    SPLIT_COLUMN,
    SPLIT_DAY,
    TOP_CATEGORIES,
    _to_category,
    feature_sql,
    null_flag_names,
    time_split,
)


def test_null_flags_have_no_duplicates():
    assert len(NULL_FLAG_COLUMNS) == len(set(NULL_FLAG_COLUMNS))


def test_flag_names_are_valid_identifiers():
    for name in null_flag_names():
        assert name.isidentifier(), name


def test_feature_sql_defines_all_flags():
    sql = feature_sql()
    for name in null_flag_names():
        assert f"AS {name}" in sql


def test_feature_sql_defines_derived_features():
    sql = feature_sql()
    for name in ("amt_decimals", "card1_freq_log", "card1_gap_sec"):
        assert f"AS {name}" in sql


def test_feature_sql_runs_on_duckdb():
    """생성된 SQL 이 실제로 실행되는지 확인한다.

    컬럼명 오타나 문법 오류를 웨어하우스 없이 잡는다.
    """
    con = duckdb.connect(":memory:")
    cols = ", ".join(
        f'NULL::INTEGER AS "{c}"'
        for c in {*NULL_FLAG_COLUMNS, "card1", "TransactionDT"}
    )
    con.execute(f"CREATE TABLE txn AS SELECT 100.0 AS TransactionAmt, {cols}")
    row = con.execute(f"SELECT {feature_sql()} FROM txn").df()
    assert len(row) == 1
    assert row["amt_decimals"].iloc[0] == 0  # 100.0 은 정수


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(100.0, 0), (100.50, 2), (100.123, 3), (57.95, 2)],
)
def test_amt_decimals_classification(amount, expected):
    con = duckdb.connect(":memory:")
    got = con.execute(f"""
        SELECT CASE
                 WHEN {amount} = FLOOR({amount}) THEN 0
                 WHEN ROUND({amount}, 2) = {amount} THEN 2
                 ELSE 3
               END
    """).fetchone()[0]
    assert got == expected


def test_to_category_caps_cardinality():
    s = pd.Series([f"v{i}" for i in range(TOP_CATEGORIES + 20)])
    out = _to_category(s)
    assert len(out.cat.categories) <= TOP_CATEGORIES + 1  # +1 은 '기타'
    assert "기타" in out.cat.categories


def test_to_category_keeps_small_cardinality():
    s = pd.Series(["a", "b", "a", "c"])
    out = _to_category(s)
    assert set(out.cat.categories) == {"a", "b", "c"}
    assert "기타" not in out.cat.categories


def test_time_split_uses_split_day():
    df = pd.DataFrame({SPLIT_COLUMN: [1, SPLIT_DAY, SPLIT_DAY + 1, 182]})
    train, valid = time_split(df)
    assert train[SPLIT_COLUMN].max() == SPLIT_DAY
    assert valid[SPLIT_COLUMN].min() == SPLIT_DAY + 1
    assert len(train) + len(valid) == len(df)


def test_categorical_columns_are_unique():
    assert len(CATEGORICAL_COLUMNS) == len(set(CATEGORICAL_COLUMNS))
