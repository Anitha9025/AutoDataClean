"""
Robust file handling utilities for the Smart Data Cleaning System.
Handles CSV, Excel, and JSON files with optimization for large datasets.
"""

import json
import logging
import os
import psutil
from pathlib import Path
from typing import Optional, Union, Iterator, Dict, Any, List, Tuple
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import (
    MAX_FILE_SIZE, CHUNK_SIZE, MEMORY_THRESHOLD, SUPPORTED_FORMATS,
    CSV_CHUNK_PARAMS, EXCEL_PARAMS, JSON_PARAMS, PANDAS_OPTIONS
)
from .exceptions import (
    FileHandlingError, UnsupportedFileFormatError, FileSizeError,
    FileCorruptionError, InsufficientMemoryError
)

# Configure pandas for optimal performance
for option, value in PANDAS_OPTIONS.items():
    pd.set_option(option, value)

# Suppress pandas warnings for large file operations
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


class FileHandler:
    """
    Robust file handler for CSV, Excel, and JSON files with large file optimization.
    """
    
    def __init__(self, memory_threshold: float = MEMORY_THRESHOLD):
        """
        Initialize FileHandler with memory monitoring.
        
        Args:
            memory_threshold: Maximum memory usage threshold (0.0 to 1.0)
        """
        self.memory_threshold = memory_threshold
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for file operations."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _check_memory_usage(self) -> None:
        """Check current memory usage and raise exception if threshold exceeded."""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            raise InsufficientMemoryError(
                f"Memory usage ({memory_percent:.1%}) exceeds threshold "
                f"({self.memory_threshold:.1%})"
            )
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate file exists, size, and format.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            FileHandlingError: If file doesn't exist or is inaccessible
            UnsupportedFileFormatError: If file format is not supported
            FileSizeError: If file size exceeds maximum allowed size
        """
        if not file_path.exists():
            raise FileHandlingError(f"File not found", str(file_path))
        
        if not file_path.is_file():
            raise FileHandlingError(f"Path is not a file", str(file_path))
        
        # Check file format
        file_extension = file_path.suffix.lower()
        if file_extension not in SUPPORTED_FORMATS:
            raise UnsupportedFileFormatError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
                str(file_path)
            )
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise FileSizeError(
                f"File size ({file_size / (1024**3):.2f} GB) exceeds "
                f"maximum allowed size ({MAX_FILE_SIZE / (1024**3):.2f} GB)",
                str(file_path)
            )
        
        self.logger.info(f"File validation passed: {file_path.name} "
                        f"({file_size / (1024**2):.2f} MB)")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        self._validate_file(file_path)
        
        file_stat = file_path.stat()
        file_info = {
            'name': file_path.name,
            'path': str(file_path),
            'size_bytes': file_stat.st_size,
            'size_mb': file_stat.st_size / (1024 ** 2),
            'size_gb': file_stat.st_size / (1024 ** 3),
            'extension': file_path.suffix.lower(),
            'modified_time': file_stat.st_mtime,
        }
        
        # Get estimated row count for structured files
        try:
            if file_path.suffix.lower() == '.csv':
                file_info['estimated_rows'] = self._estimate_csv_rows(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                file_info['sheets'] = self._get_excel_sheets(file_path)
        except Exception as e:
            self.logger.warning(f"Could not get additional file info: {e}")
        
        return file_info
    
    def _estimate_csv_rows(self, file_path: Path) -> int:
        """Estimate number of rows in a CSV file by sampling."""
        try:
            # Sample first 1MB to estimate average row length
            sample_size = min(1024 * 1024, file_path.stat().st_size)
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(sample_size)
                sample_rows = sample.count('\n')
                
            if sample_rows == 0:
                return 0
                
            # Estimate total rows based on file size
            total_size = file_path.stat().st_size
            estimated_rows = int((total_size / sample_size) * sample_rows)
            
            return max(0, estimated_rows - 1)  # Subtract header row
            
        except Exception:
            return -1  # Unknown
    
    def _get_excel_sheets(self, file_path: Path) -> List[str]:
        """Get list of sheet names from Excel file."""
        try:
            with pd.ExcelFile(file_path) as xl_file:
                return xl_file.sheet_names
        except Exception:
            return []
    
    def read_file(self, 
                  file_path: Union[str, Path],
                  use_chunks: Optional[bool] = None,
                  chunk_size: Optional[int] = None,
                  **kwargs) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Read file with automatic format detection and optimization.
        
        Args:
            file_path: Path to the file
            use_chunks: Force chunked reading (auto-detect if None)
            chunk_size: Size of chunks for large files
            **kwargs: Additional parameters for pandas readers
            
        Returns:
            DataFrame or DataFrame iterator for large files
        """
        file_path = Path(file_path)
        self._validate_file(file_path)
        self._check_memory_usage()
        
        file_extension = file_path.suffix.lower()
        file_size_mb = file_path.stat().st_size / (1024 ** 2)
        
        # Auto-determine if chunking is needed (files > 100MB)
        if use_chunks is None:
            use_chunks = file_size_mb > 100
        
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
            
        self.logger.info(f"Reading {file_path.name} ({file_size_mb:.2f} MB)"
                        f"{' in chunks' if use_chunks else ''}")
        
        try:
            if file_extension == '.csv':
                return self._read_csv(file_path, use_chunks, chunk_size, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                return self._read_excel(file_path, **kwargs)
            elif file_extension == '.json':
                return self._read_json(file_path, **kwargs)
        except Exception as e:
            raise FileCorruptionError(
                f"Failed to read file - file may be corrupted: {str(e)}",
                str(file_path)
            )
    
    def _read_csv(self, 
                  file_path: Path, 
                  use_chunks: bool, 
                  chunk_size: int,
                  **kwargs) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Read CSV file with chunking support."""
        # Merge default CSV parameters with user parameters
        csv_params = {**CSV_CHUNK_PARAMS}
        csv_params.update(kwargs)
        
        if use_chunks:
            csv_params['chunksize'] = chunk_size
            self.logger.info(f"Reading CSV in chunks of {chunk_size:,} rows")
            return pd.read_csv(file_path, **csv_params)
        else:
            # Remove chunksize for full file reading
            csv_params.pop('chunksize', None)
            return pd.read_csv(file_path, **csv_params)
    
    def _read_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read Excel file with optimization."""
        excel_params = {**EXCEL_PARAMS}
        excel_params.update(kwargs)
        
        # For large Excel files, warn about potential memory usage
        file_size_mb = file_path.stat().st_size / (1024 ** 2)
        if file_size_mb > 50:
            self.logger.warning(f"Large Excel file ({file_size_mb:.1f} MB) - "
                              "consider converting to CSV for better performance")
        
        return pd.read_excel(file_path, **excel_params)
    
    def _read_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read JSON file with error handling."""
        json_params = {**JSON_PARAMS}
        json_params.update(kwargs)
        
        try:
            # Try pandas read_json first
            return pd.read_json(file_path, **json_params)
        except Exception:
            # Fallback to manual JSON reading for complex structures
            self.logger.info("Falling back to manual JSON parsing")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to normalize nested JSON
                return pd.json_normalize(data)
            else:
                raise FileCorruptionError("Unsupported JSON structure", str(file_path))
    
    def process_large_file_in_chunks(self,
                                   file_path: Union[str, Path],
                                   processor_func: callable,
                                   chunk_size: Optional[int] = None,
                                   show_progress: bool = True) -> List[Any]:
        """
        Process large files chunk by chunk with progress tracking.
        
        Args:
            file_path: Path to the file
            processor_func: Function to process each chunk
            chunk_size: Size of each chunk
            show_progress: Whether to show progress bar
            
        Returns:
            List of results from processing each chunk
        """
        file_path = Path(file_path)
        chunk_size = chunk_size or CHUNK_SIZE
        
        # Get file info for progress tracking
        file_info = self.get_file_info(file_path)
        estimated_chunks = max(1, file_info.get('estimated_rows', 0) // chunk_size)
        
        chunk_iterator = self.read_file(file_path, use_chunks=True, chunk_size=chunk_size)
        
        results = []
        chunk_num = 0
        
        # Set up progress bar
        progress_bar = None
        if show_progress and estimated_chunks > 1:
            progress_bar = tqdm(
                total=estimated_chunks,
                desc=f"Processing {file_path.name}",
                unit="chunks"
            )
        
        try:
            for chunk in chunk_iterator:
                self._check_memory_usage()
                
                # Process chunk
                result = processor_func(chunk, chunk_num)
                results.append(result)
                
                chunk_num += 1
                if progress_bar:
                    progress_bar.update(1)
                    
        finally:
            if progress_bar:
                progress_bar.close()
        
        self.logger.info(f"Processed {chunk_num} chunks from {file_path.name}")
        return results
    
    def save_dataframe(self,
                      df: pd.DataFrame,
                      output_path: Union[str, Path],
                      format_type: Optional[str] = None,
                      **kwargs) -> None:
        """
        Save DataFrame with format auto-detection and optimization.
        
        Args:
            df: DataFrame to save
            output_path: Path for output file
            format_type: Force specific format (auto-detect if None)
            **kwargs: Additional parameters for pandas writers
        """
        output_path = Path(output_path)
        
        # Auto-detect format from extension if not specified
        if format_type is None:
            format_type = output_path.suffix.lower()
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving DataFrame to {output_path.name} "
                        f"({df.shape[0]:,} rows, {df.shape[1]} columns)")
        
        try:
            if format_type == '.csv':
                df.to_csv(output_path, index=False, **kwargs)
            elif format_type in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False, **kwargs)
            elif format_type == '.json':
                df.to_json(output_path, orient='records', **kwargs)
            else:
                raise UnsupportedFileFormatError(
                    f"Unsupported output format: {format_type}",
                    str(output_path)
                )
                
        except Exception as e:
            raise FileHandlingError(
                f"Failed to save file: {str(e)}",
                str(output_path)
            )


def create_sample_processor() -> callable:
    """
    Create a sample processor function for demonstration.
    
    Returns:
        Sample processor function
    """
    def sample_processor(chunk: pd.DataFrame, chunk_num: int) -> Dict[str, Any]:
        """Sample processor that returns basic chunk statistics."""
        return {
            'chunk_number': chunk_num,
            'rows': len(chunk),
            'columns': len(chunk.columns),
            'memory_usage_mb': chunk.memory_usage(deep=True).sum() / (1024 ** 2),
            'null_values': chunk.isnull().sum().sum()
        }
    
    return sample_processor