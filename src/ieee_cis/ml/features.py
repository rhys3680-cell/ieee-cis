"""모델 학습용 피처 만들기

설계 근거는 notebooks/rhys/02_feature_signals.ipynb
"""

import duckdb
import pandas as pd

#: 시간 기반 분할 적용 train 1~182 일차의 80% 지점.
#: 학습 1~141 일차, 검증 142~182일차.
#:
#: 랜덤 분할을 쓰면 시계열 데이터 특성 상에 데이터 누수가 발생할 수도 있기 때문에 사용 X
#:
#: 참고: 랜덤 분할과 시간 기반 분할을 적용해보았을 때 둘의 차이가 2% 정도로 크지 않았지만 특성 상 반영

SPLIT_DAY = 141

#: 결측 여부 자체가 의미가 있는 컬럼
#:
#: 후보 49개 중 리프트가 3.5 이상이면서, 결측률과 리프트가 같은 중복 그룹 중에서 대표만 남김
#:
# 참고: D12 / D13 / D14 는 결측률이 미세하게 달라 중복그룹으로 되지 않음
NULL_FLAG_COLUMNS = (
    "D7",  # 93.4% 결측, 5.52x — 값이 있을 때 위험
    "addr1",  # 11.1% 결측, 4.79x — 없을 때 위험. addr2 와 동일 그룹
    "D12",  # 89.0% 결측, 4.73x
    "D14",  # 89.5% 결측, 4.56x
    "D13",  # 89.5% 결측, 4.22x
    "D6",  # 87.6% 결측, 4.21x
    "D9",  # 87.3% 결측, 4.20x — D8/id_09/id_10 과 동일 그룹
    "id_03",  # 88.8% 결측, 4.15x — id_04 와 동일 그룹
    "R_emaildomain",  # 76.8% 결측, 3.93x
    "id_13",  # 78.4% 결측, 3.82x
    "id_02",  # 76.1% 결측, 3.79x — id_15/35~38 등 9 개 그룹의 대표
    "id_31",  # 76.2% 결측, 3.78x
    "id_05",  # 76.8% 결측, 3.76x — id_06 과 동일 그룹
    "id_12",  # 75.6% 결측, 3.75x — id_01 과 동일 그룹
    "id_17",  # 77.4% 결측, 3.73x
    "id_19",  # 76.4% 결측, 3.71x — id_20 과 동일 그룹
)

#: LightGBM 에 Category 로 넘길 컬럼.
CATEGORICAL_COLUMNS = (
    "DeviceInfo",
    "id_33",
    "id_31",
    "id_30",
    "P_emaildomain",
    "R_emaildomain",
    "ProductCD",
    "card6",
    "card4",
    "id_34",
    "M4",
    "id_15",
    "id_23",
    "id_12",
    "id_16",
    "id_27",
    "id_29",
    "id_28",
    "DeviceType",
)

#: 고카디널리티 범주형에서 남길 상위 카테고리 수. 나머지는 '기타'.
#: DeviceInfo 2,799 / id_33 461 / id_31 172
TOP_CATEGORIES = 50

#: 피처에서 제외할 컬럼.
#: TransactionDT 시간 정보를 담고 있음.
EXCLUDE_COLUMNS = (
    "TransactionID",
    "isFraud",
    "dataset_split",
    "TransactionDT",
)

#: 분할용 보조 컬럼. 학습 직전에 드롭
SPLIT_COLUMN = "_day_index"


def _flag_name(column: str) -> str:
    return f"is_null_{column.lower()}"


def null_flag_names() -> list[str]:
    return [_flag_name(c) for c in NULL_FLAG_COLUMNS]


def feature_sql() -> str:
    """파생 피처 생성 SQL"""
    parts = [
        # 금액 소수 자릿수. 3 자리 이상 11.72% vs 2 자리 1.13% (10배).
        # 환전일 가능성
        # 소수부 유무는 기각
        """CASE
            WHEN TransactionAmt = FLOOR(TransactionAmt) THEN 0
            WHEN ROUND(TransactionAmt, 2) = TransactionAmt THEN 2
            ELSE 3
           END::UTINYINT AS amt_decimals""",
        # 카드 사용 빈도. 1 건 카드 5.21% vs 6~50 건 2.53%의 U 자형.
        # 중앙값 5, p90 63, 최대 28,015로 로그를 씌운다.
        "LOG(COUNT(*) OVER (PARTITION BY card1) + 1) AS card1_freq_log",
        # 동일 카드 직전 거래 경과 초. 1 분 이내 6.38% vs 1일 이상 2.40%.
        # 첫 거래(1.56%)는 NULL 로 둔다. 0으로 채우면 "0 초 전 거래"가 된다.
        """TransactionDT - LAG(TransactionDT) OVER (
                PARTITION BY card1 ORDER BY TransactionDT
            ) AS card1_gap_sec""",
    ]
    parts += [f'("{c}" IS NULL) AS {_flag_name(c)}' for c in NULL_FLAG_COLUMNS]
    return ",\n           ".join(parts)


def load(con: duckdb.DuckDBPyConnection, split: str = "train"):
    """웨어하우스에서 피처를 만들어 DataFrame 으로 반환한다.

    Args:
        con: warehouse.duckdb 커넥션 (read_only 권장).
        split: 'train' 또는 'test'.
    """
    df = con.execute(f"""
        SELECT * EXCLUDE ({", ".join(EXCLUDE_COLUMNS)}),
                TransactionID,
                isFraud,
                day_index AS {SPLIT_COLUMN},
                {feature_sql()}
        FROM txn
        WHERE dataset_split = '{split}'
        ORDER BY TransactionDT
    """).df()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = _to_category(df[col])
    return df


def _to_category(series: pd.Series):
    """범주형으로 변환한다. 고카디널리티는 상위 N 개만 남긴다."""
    if series.nunique() > TOP_CATEGORIES:
        top = series.value_counts().nlargest(TOP_CATEGORIES).index
        series = series.where(series.isin(top), "기타")
    return series.astype("category")


def time_split(df):
    """시간 기반으로 학습/검증을 나눈다.

    Returns:
        (train_df, valid_df)
    """
    is_train = df[SPLIT_COLUMN] <= SPLIT_DAY
    return df[is_train].copy(), df[~is_train].copy()
