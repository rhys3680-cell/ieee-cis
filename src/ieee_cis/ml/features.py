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

#: 결측 여부 플래그. **v1 학습 결과 전부 제거했다.**
#:
#: 단변량 조사에서는 강한 신호였다(D7 리프트 5.52x, addr1 4.79x).
#: 그러나 v1 importance 에서 16 개 중 최고가 274 위, 8 개는 gain 이
#: 정확히 0 으로 트리가 한 번도 쓰지 않았다. D7 은 단변량 최상위였으나
#: importance 408 위다.
#:
#: 원인은 LightGBM 이 NULL 을 자체 분기로 처리하기 때문이다. 원본 컬럼을
#: 넣으면 트리가 "NULL 이면 왼쪽" 분기를 알아서 만들므로 별도 플래그는
#: 완전한 중복이다.
#:
#: 단변량 신호가 다변량 기여로 이어지지 않은 사례다. 근거는
#: data/models/importance_v1.json 에 남아 있다.
NULL_FLAG_COLUMNS = ()

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
    "day_index",
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
