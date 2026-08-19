"""TransactionDT 에서 시간 관련 피처 파생.

원본은 미상의 기준 시각으로부터의 초 오프셋만 제공한다. 실제 날짜와 시각은 알 수 없다.

따라서 실제 시각으로 변환하지 않고 상대 지표만 만든다.

train 590,540 건 집계 결과:
  시간대(hour%24) 사기율 2.29~10.61%, 4.64배 -> 추가
"""

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400

#: 파생 컬럼명. build_warehouse와 semantic 레지스트리가 공유한다.
DERIVED_COLUMNS = ("hour_slot", "day_index")


def hour_slot(transaction_dt: int) -> int:
    """24시간 주기 안에서의 슬롯 (0~23).

    실제 시각이 아니다.
    """
    return (transaction_dt // SECONDS_PER_HOUR) % 24


def day_index(transaction_dt: int) -> int:
    """오프셋을 일 단위로 나눈 값.

    실제 데이터 범위는 train 1~182, test 213~395 다. 오프셋 0 인 거래는
    존재하지 않으므로 사실상 1 부터 시작.

    시간 기반 학습/검증 분할과 추이 표시에 사용. 화면에는 그대로 사용.
    """
    return transaction_dt // SECONDS_PER_DAY


def derived_columns_sql(source: str = "TransactionDT") -> str:
    """파생 컬럼을 만드는 SQL"""
    return (
        f"(({source} // {SECONDS_PER_HOUR}) % 24)::TINYINT AS hour_slot, "
        f"({source} // {SECONDS_PER_DAY})::SMALLINT AS day_index"
    )
