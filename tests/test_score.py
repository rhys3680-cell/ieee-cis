"""점수 적재 테스트.

실제 점수화는 110 만 행에 1 분 40 초가 걸리므로 테스트하지 않는다.
컬럼 정렬과 검증 로직만 확인한다.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from ieee_cis.ml.score import _predict, verify


class FakeBooster:
    """예측 대신 입력 컬럼 순서를 기록하는 대역."""

    def __init__(self, names):
        self._names = names
        self.seen_columns = None

    def feature_name(self):
        return self._names

    def predict(self, x):
        self.seen_columns = list(x.columns)
        return np.full(len(x), 0.5)


def test_predict_reorders_columns():
    """학습 때와 컬럼 순서가 다르면 조용히 틀린 점수가 나온다."""
    booster = FakeBooster(["a", "b", "c"])
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "c": [3, 3],
            "a": [1, 1],
            "b": [2, 2],  # 순서가 뒤바뀜
        }
    )
    out = _predict(booster, df)
    assert booster.seen_columns == ["a", "b", "c"]
    assert list(out.columns) == ["TransactionID", "score"]


def test_predict_drops_label_and_id():
    booster = FakeBooster(["a"])
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "isFraud": [0],
            "_day_index": [5],
            "a": [1],
        }
    )
    _predict(booster, df)
    assert booster.seen_columns == ["a"]


def test_predict_rejects_missing_feature():
    booster = FakeBooster(["a", "missing"])
    df = pd.DataFrame({"TransactionID": [1], "a": [1]})
    with pytest.raises(ValueError, match="missing"):
        _predict(booster, df)


def _setup(con, scores, txn_rows=3):
    con.execute(f"CREATE TABLE txn AS SELECT * FROM range(1, {txn_rows + 1})")
    con.execute(
        "CREATE TABLE txn_scores (TransactionID INTEGER, score DOUBLE, model_version VARCHAR)"
    )
    for i, s in enumerate(scores, 1):
        con.execute("INSERT INTO txn_scores VALUES (?, ?, 'v1')", [i, s])


def test_verify_accepts_valid():
    con = duckdb.connect(":memory:")
    _setup(con, [0.1, 0.5, 0.9])
    verify(con, "v1")


def test_verify_rejects_row_count_mismatch():
    con = duckdb.connect(":memory:")
    _setup(con, [0.1, 0.5], txn_rows=3)
    with pytest.raises(ValueError, match="행 수"):
        verify(con, "v1")


def test_verify_rejects_out_of_range_score():
    con = duckdb.connect(":memory:")
    _setup(con, [0.1, 0.5, 1.5])
    with pytest.raises(ValueError, match="점수 범위"):
        verify(con, "v1")
