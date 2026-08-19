"""프로젝트 경로."""

from pathlib import Path

#: 저장소 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "ieee-fraud-detection"
WAREHOUST_PATH = PROJECT_ROOT / "data" / "warehouse" / "warehouse.duckdb"
