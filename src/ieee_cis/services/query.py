"""웨어하우스 조회용 커넥션."""

import duckdb

from ieee_cis.config import WAREHOUSE_PATH

#: 큰 집계에서 OOM 을 막는다.
MEMORY_LIMIT = "2GB"


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """웨어하우스 커넥션을 연다.

    앱은 항상 read_only 로 연다. 웨어하우스는 ETL 이 만드는 산출물이며
    앱이 수정하지 않는다. 분석가 판정 같은 가변 상태는 ops.sqlite 로 간다.
    """
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"웨어하우스가 없습니다: {WAREHOUSE_PATH}\n"
            "먼저 생성하세요: uv run python -m ieee_cis.etl.build_warehouse"
        )
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=read_only)
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    return con