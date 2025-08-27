"""
Main entry point for Smart Data Cleaning System - Phase 1
Demonstrates bulletproof file handling capabilities.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.file_utils import FileHandler, create_sample_processor
from src.config import ensure_directories
from src.exceptions import DataCleaningError


def setup_logging() -> None:
    """Set up logging for the application."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'smart_data_cleaning.log'),
            logging.StreamHandler()
        ]
    )


def verify_setup() -> bool:
    """Verify that the system is properly set up."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Test file handler initialization
        FileHandler()
        
        # Check required modules
        import pandas as pd
        import numpy as np
        import psutil
        import tqdm
        
        print("✅ All dependencies available")
        print("✅ Directory structure created")
        print("✅ System ready")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False


def main():
    """Main function demonstrating file handling capabilities."""
    
    print("🎯 Smart Data Cleaning System - Phase 1")
    print("=" * 50)
    
    # Set up logging
    setup_logging()
    
    # Verify setup
    print("\n🔧 Verifying system setup...")
    if not verify_setup():
        print("\n❌ Setup verification failed. Please fix the issues above.")
        return
    
    # Initialize file handler
    try:
        file_handler = FileHandler()
        print("\n✅ File handler initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize file handler: {e}")
        return
    
    # Interactive file selection
    while True:
        print("\n" + "=" * 50)
        print("📋 OPTIONS:")
        print("1. 📊 Analyze a file")
        print("2. 🔄 Process file in chunks (demo)")
        print("3. 🧪 Run system tests")
        print("4. ℹ️  Show system info")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            analyze_file(file_handler)
        elif choice == '2':
            process_file_demo(file_handler)
        elif choice == '3':
            run_tests()
        elif choice == '4':
            show_system_info()
        elif choice == '5':
            print("\n👋 Thanks for using Smart Data Cleaning System!")
            print("Phase 1 complete - ready for Phase 2 development!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")


def analyze_file(file_handler: FileHandler) -> None:
    """Analyze a file and display comprehensive information."""
    
    file_path = input("\n📁 Enter file path: ").strip().strip('"\'')
    
    if not file_path:
        print("❌ No file path provided.")
        return
    
    try:
        # Get file information
        print("\n📊 Analyzing file...")
        file_info = file_handler.get_file_info(file_path)
        
        # Display file information
        print(f"\n✅ FILE ANALYSIS RESULTS:")
        print(f"  📁 Name: {file_info['name']}")
        print(f"  📏 Size: {file_info['size_mb']:.2f} MB ({file_info['size_bytes']:,} bytes)")
        print(f"  📄 Format: {file_info['extension']}")
        
        if 'estimated_rows' in file_info and file_info['estimated_rows'] > 0:
            print(f"  📊 Estimated rows: {file_info['estimated_rows']:,}")
        
        if 'sheets' in file_info and file_info['sheets']:
            print(f"  📋 Excel sheets: {', '.join(file_info['sheets'])}")
        
        # Try to read file preview
        print("\n🔍 Reading file preview...")
        
        if file_info['size_mb'] > 100:
            print("  ⚠️  Large file detected - using chunked reading")
            chunk_iterator = file_handler.read_file(file_path, use_chunks=True, chunk_size=1000)
            preview_df = next(iter(chunk_iterator))
            print(f"  📝 Showing first 1,000 rows of ~{file_info.get('estimated_rows', '?'):,} total rows")
        else:
            preview_df = file_handler.read_file(file_path, use_chunks=False)
            print(f"  📝 Loaded complete file: {len(preview_df):,} rows")
        
        # Display preview
        print(f"\n📈 DATA SUMMARY:")
        print(f"  🔢 Shape: {preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns")
        print(f"  📋 Columns: {list(preview_df.columns)}")
        print(f"\n📊 Data types:")
        for col, dtype in preview_df.dtypes.items():
            print(f"    {col}: {dtype}")
        
        # Show basic statistics
        numeric_cols = preview_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\n📊 Numeric column statistics:")
            stats = preview_df[numeric_cols].describe()
            print(stats.to_string())
        
        print(f"\n🔍 FIRST 5 ROWS:")
        print(preview_df.head().to_string(max_cols=10))
        
        # Memory usage and data quality
        memory_mb = preview_df.memory_usage(deep=True).sum() / (1024 ** 2)
        null_count = preview_df.isnull().sum().sum()
        
        print(f"\n💾 MEMORY & QUALITY:")
        print(f"  💾 Memory usage: {memory_mb:.2f} MB")
        print(f"  ❓ Null values: {null_count:,}")
        print(f"  🎯 Data completeness: {(1 - null_count/preview_df.size)*100:.1f}%")
        
    except DataCleaningError as e:
        print(f"❌ Error: {e}")
        logging.error(f"File analysis error: {e}")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        logging.error(f"Unexpected error in file analysis: {e}")


def process_file_demo(file_handler: FileHandler) -> None:
    """Demonstrate chunk processing capabilities."""
    
    file_path = input("\n📁 Enter file path for chunk processing demo: ").strip().strip('"\'')
    
    if not file_path:
        print("❌ No file path provided.")
        return
    
    try:
        print("\n🔄 Processing file in chunks...")
        
        # Get file info first
        file_info = file_handler.get_file_info(file_path)
        print(f"📊 File: {file_info['name']} ({file_info['size_mb']:.2f} MB)")
        
        # Create sample processor
        processor = create_sample_processor()
        
        # Process file
        chunk_size = 5000  # Process 5000 rows at a time
        print(f"🔄 Processing in chunks of {chunk_size:,} rows...")
        
        results = file_handler.process_large_file_in_chunks(
            file_path=file_path,
            processor_func=processor,
            chunk_size=chunk_size,
            show_progress=True
        )
        
        # Display results
        print(f"\n📊 CHUNK PROCESSING RESULTS:")
        print(f"  📦 Total chunks processed: {len(results)}")
        
        total_rows = sum(r['rows'] for r in results)
        total_memory = sum(r['memory_usage_mb'] for r in results)
        total_nulls = sum(r['null_values'] for r in results)
        
        print(f"  📊 Total rows processed: {total_rows:,}")
        print(f"  💾 Total memory used: {total_memory:.2f} MB")
        print(f"  ❓ Total null values found: {total_nulls:,}")
        print(f"  📈 Average chunk size: {total_rows/len(results):.0f} rows")
        
        # Show chunk details
        if len(results) <= 10:
            print(f"\n📋 DETAILED CHUNK BREAKDOWN:")
            for result in results:
                print(f"    Chunk {result['chunk_number']}: "
                      f"{result['rows']:,} rows, "
                      f"{result['memory_usage_mb']:.1f} MB, "
                      f"{result['null_values']:,} nulls")
        else:
            print(f"\n📋 SAMPLE CHUNK DETAILS (first 5):")
            for result in results[:5]:
                print(f"    Chunk {result['chunk_number']}: "
                      f"{result['rows']:,} rows, "
                      f"{result['memory_usage_mb']:.1f} MB, "
                      f"{result['null_values']:,} nulls")
            print(f"    ... and {len(results)-5} more chunks")
        
    except DataCleaningError as e:
        print(f"❌ Error: {e}")
        logging.error(f"Chunk processing error: {e}")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        logging.error(f"Unexpected error in chunk processing: {e}")


def run_tests() -> None:
    """Run system tests."""
    print("\n🧪 RUNNING SYSTEM TESTS...")
    print("=" * 30)
    
    try:
        import unittest
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
            
    except ImportError:
        print("❌ Tests not available - missing test files")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")


def show_system_info() -> None:
    """Show system information and capabilities."""
    print("\n🖥️  SYSTEM INFORMATION:")
    print("=" * 30)
    
    try:
        import pandas as pd
        import numpy as np
        import psutil
        
        # System info
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        print(f"📊 System Resources:")
        print(f"  💾 Memory: {memory.total / (1024**3):.1f} GB total, "
              f"{memory.available / (1024**3):.1f} GB available "
              f"({memory.percent:.1f}% used)")
        print(f"  💽 Disk: {disk.total / (1024**3):.1f} GB total, "
              f"{disk.free / (1024**3):.1f} GB free "
              f"({(disk.used/disk.total)*100:.1f}% used)")
        
        # Library versions
        print(f"\n📚 Library Versions:")
        print(f"  🐼 Pandas: {pd.__version__}")
        print(f"  🔢 NumPy: {np.__version__}")
        print(f"  📊 PSUtil: {psutil.__version__}")
        
        # Capabilities
        from src.config import SUPPORTED_FORMATS, MAX_FILE_SIZE, CHUNK_SIZE
        
        print(f"\n🎯 System Capabilities:")
        print(f"  📄 Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        print(f"  📏 Max file size: {MAX_FILE_SIZE / (1024**3):.1f} GB")
        print(f"  📦 Default chunk size: {CHUNK_SIZE:,} rows")
        print(f"  🧠 Memory threshold: {psutil.virtual_memory().percent:.1f}%")
        
        # Directory structure
        from src.config import BASE_DIR, INPUT_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR
        
        print(f"\n📁 Directory Structure:")
        dirs = [
            ("Base", BASE_DIR),
            ("Input", INPUT_DIR), 
            ("Output", OUTPUT_DIR),
            ("Temp", TEMP_DIR),
            ("Logs", LOGS_DIR)
        ]
        
        for name, path in dirs:
            exists = "✅" if path.exists() else "❌"
            print(f"  {exists} {name}: {path}")
        
    except Exception as e:
        print(f"❌ Error getting system info: {e}")


if __name__ == "__main__":
    main()