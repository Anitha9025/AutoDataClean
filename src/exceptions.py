
"""Custom exceptions for the Smart Data Cleaning System."""

from typing import Optional


class DataCleaningError(Exception):
    """Base exception for all data cleaning operations."""
    pass


class FileHandlingError(DataCleaningError):
    """Exception raised for file handling operations."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        if file_path:
            message = f"{message} (File: {file_path})"
        super().__init__(message)


class UnsupportedFileFormatError(FileHandlingError):
    """Exception raised when file format is not supported."""
    pass


class FileSizeError(FileHandlingError):
    """Exception raised when file size exceeds limits."""
    pass


class FileCorruptionError(FileHandlingError):
    """Exception raised when file appears to be corrupted."""
    pass


class InsufficientMemoryError(DataCleaningError):
    """Exception raised when system doesn't have enough memory."""
    pass


class ProfilingError(DataCleaningError):
    """Exception raised during data profiling operations."""
    pass


class DataQualityError(DataCleaningError):
    """Exception raised during data quality assessment."""
    pass
