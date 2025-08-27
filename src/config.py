"""Configuration settings for the Smart Data Cleaning System."""

import os
from pathlib import Path
from typing import Dict, Any

# File size limits (in bytes)
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB
CHUNK_SIZE = 10000  # Default chunk size for processing large files
MEMORY_THRESHOLD = 0.8  # 80% memory usage threshold

# Supported file formats
SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.json'}

# Directory structure
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TEMP_DIR = DATA_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

# Pandas optimization settings
PANDAS_OPTIONS: Dict[str, Any] = {
    'display.max_columns': None,
    'display.max_rows': 100,
    'display.width': None,
    'display.max_colwidth': 50
}

# CSV reading parameters for large files
CSV_CHUNK_PARAMS = {
    'chunksize': CHUNK_SIZE,
    'low_memory': False,
    'engine': 'c',  # Use C engine for speed
}

# Excel reading parameters
EXCEL_PARAMS = {
    'engine': 'openpyxl',  # For .xlsx files
}

# JSON reading parameters
JSON_PARAMS = {
    'lines': False,  # Set to True for JSONL files
}

def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)