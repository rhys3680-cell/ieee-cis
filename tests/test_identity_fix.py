"""identity 컬럼명 정규화 테스트."""

from __future__ import annotations

import pytest

from ieee_cis.etl.identity_fix import (
    normalize_identity_columns,
    verify_identity_schema,
)

# 실제 CSV 헤더
TRAIN_HEADER = [
    "TransactionID",
    *[f"id_{i:02d}" for i in range(1, 39)],
    "DeviceType",
    "DeviceInfo",
]
TEST_HEADER = [
    "TransactionID",
    *[f"id-{i:02d}" for i in range(1, 39)],
    "DeviceType",
    "DeviceInfo",
]


def test_test_header_becomes_train_header() -> None:
    """38개 컬럼 전부 변환되고 나머지는 그대로, 순서도 보존."""
    assert normalize_identity_columns(TEST_HEADER) == TRAIN_HEADER


def test_train_header_unchanged() -> None:
    """train 에 적용해도 무해해야 한다 - 분기 없이 항상 호출하기 위함."""
    assert normalize_identity_columns(TRAIN_HEADER) == TRAIN_HEADER


@pytest.mark.parametrize("name", ["D-1", "id-1", "id-001", "xid-01", "card1"])
def test_does_not_touch_unrelated_columns(name: str) -> None:
    assert normalize_identity_columns([name]) == [name]


def test_verify_accepts_normalized() -> None:
    verify_identity_schema(normalize_identity_columns(TEST_HEADER))


def test_verify_rejects_unnormalized() -> None:
    """정규화 없이 병합하려 하면 반드시 에러 발생시켜야 한다."""
    with pytest.raises(ValueError, match="하이픈"):
        verify_identity_schema(TEST_HEADER)


def test_verify_rejects_missing_id_column() -> None:
    with pytest.raises(ValueError, match="38"):
        verify_identity_schema([c for c in TRAIN_HEADER if c != "id_20"])


@pytest.mark.parametrize("missing", ["TransactionID", "DeviceType", "DeviceInfo"])
def test_verify_rejects_missing_required(missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        verify_identity_schema([c for c in TRAIN_HEADER if c != missing])
