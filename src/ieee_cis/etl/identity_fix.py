"""test_identity.csv의 하이픈 컬럼명을 train과 동일한 언더스코어로 정규화

train_identity.csv -> id_01 ... id_38 (언더스코어)
test_identity.csv  -> id-01 ... id-38 (하이픈)

"""

import re
from collections.abc import Iterable

#: id-01 ~ id-38 형태만 매칭.
_HYPHEN_ID = re.compile(r"^id-(\d{2})$")

#: 정규화 후 identity 파일이 가져야 할 id 컬럼 개수
EXPECTED_ID_COLUMNS = 38


def normalize_identity_columns(names: Iterable[str]) -> list[str]:
    """identity 컬럼명 정규화"""
    return [_HYPHEN_ID.sub(r"id_\1", n) for n in names]


def verify_identity_schema(names: Iterable[str]) -> None:
    """정규화된 컬럼 목록 검증

    ETL에서 병합 직전에 호출

    Raises:
        ValueError: 하이픈 컬럼이 남아있거나, id 컬럼이 38개가 아니거나, 필수 컬럼이 없는 경우

    """
    cols = list(names)

    if leftover := [c for c in cols if _HYPHEN_ID.match(c)]:
        raise ValueError(f"정규화되지 않은 하이픈 컬럼이 남아있음: {leftover}")

    id_cols = [c for c in cols if re.fullmatch(r"id_\d{2}", c)]
    if len(id_cols) != EXPECTED_ID_COLUMNS:
        raise ValueError(
            f"id 컬럼이 {EXPECTED_ID_COLUMNS} 개 여야 하는데 {len(id_cols)} 개임"
        )

    for required in ("TransactionID", "DeviceType", "DeviceInfo"):
        if required not in cols:
            raise ValueError(f"필수 컬럼 누락: {required}")
