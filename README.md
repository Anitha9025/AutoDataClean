"""
# Smart Data Cleaning System - Phase 1

## 🎯 Overview

Phase 1 provides bulletproof file handling foundation for CSV, Excel, and JSON files with optimization for large datasets (1GB+).

## ✨ Features

- **Multi-format Support**: CSV, Excel (.xlsx, .xls), JSON
- **Large File Handling**: Optimized for files up to 1GB+
- **Memory Management**: Automatic memory monitoring and chunked processing
- **Error Handling**: Comprehensive error handling with clear messages
- **Progress Tracking**: Visual progress bars for long operations
- **Production Ready**: Type hints, docstrings, and logging

## 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the system:
   ```bash
   python main.py
   ```

3. Choose options to analyze files or demo chunk processing

## 📁 Project Structure

```
smart_data_cleaning/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── file_utils.py        # Core file handling
│   ├── config.py            # Configuration settings
│   └── exceptions.py        # Custom exceptions
├── data/
│   ├── input/              # Input files
│   ├── output/             # Output files
│   └── temp/               # Temporary files
├── logs/                   # Log files
├── tests/                  # Unit tests
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## 🔧 Core Components

### FileHandler Class

The main class for file operations:

```python
from src.file_utils import FileHandler

handler = FileHandler()

# Analyze file
info = handler.get_file_info("data.csv")

# Read file (auto-detects chunking for large files)
df = handler.read_file("data.csv")

# Process large files in chunks
results = handler.process_large_file_in_chunks(
    "large_data.csv", 
    processor_function,
    chunk_size=10000
)

# Save data
handler.save_dataframe(df, "output.csv")
```

### Supported Operations

- **File Validation**: Size, format, accessibility checks
- **Auto-optimization**: Automatic chunking for large files
- **Memory Monitoring**: Prevents system overload
- **Progress Tracking**: Visual feedback for long operations
- **Error Recovery**: Graceful handling of corrupted files

## 🎯 Configuration

Edit `src/config.py` to customize:

- Maximum file size (default: 1GB)
- Chunk sizes for processing
- Memory usage thresholds
- File format parameters

## 📊 Performance

- **CSV Files**: C-engine parsing with optimized parameters
- **Excel Files**: openpyxl engine with memory warnings
- **JSON Files**: Pandas optimization with fallback parsing
- **Large Files**: Automatic chunking with progress tracking

## 🧪 Example Usage

```python
# Basic file analysis
handler = FileHandler()
info = handler.get_file_info("sales_data.csv")
print(f"File has {info['estimated_rows']:,} rows")

# Read with automatic optimization
data = handler.read_file("sales_data.csv")  # Auto-chunks if large

# Custom chunk processing
def analyze_chunk(chunk, chunk_num):
    return {
        'chunk': chunk_num,
        'avg_sales': chunk['sales'].mean(),
        'null_count': chunk.isnull().sum().sum()
    }

results = handler.process_large_file_in_chunks(
    "sales_data.csv",
    analyze_chunk,
    show_progress=True
)
```

## 🔜 Next Phases

Phase 1 provides the foundation. Future phases will add:
- Data cleaning logic
- LLM integration
- RAG capabilities  
- Advanced reporting
- Web interface

## 🐛 Error Handling

The system includes comprehensive error handling:

- `FileHandlingError`: General file operation errors
- `UnsupportedFileFormatError`: Invalid file formats
- `FileSizeError`: Files exceeding size limits
- `FileCorruptionError`: Corrupted or unreadable files
- `InsufficientMemoryError`: Memory threshold exceeded

## 📈 Memory Optimization

- Automatic memory monitoring
- Chunked processing for large files
- Pandas optimization settings
- Progress tracking with minimal overhead
- Cleanup of temporary resources

Built with ❤️ for robust data processing
"""