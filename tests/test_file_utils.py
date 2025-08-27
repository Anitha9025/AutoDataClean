"""
Comprehensive unit tests for file_utils.py
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.file_utils import FileHandler, create_sample_processor
from src.exceptions import (
    FileHandlingError, UnsupportedFileFormatError, 
    FileSizeError, FileCorruptionError, InsufficientMemoryError
)
from src.config import MAX_FILE_SIZE


class TestFileHandler(unittest.TestCase):
    """Test cases for FileHandler class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.file_handler = FileHandler()
        
        # Create sample data
        self.sample_df = pd.DataFrame({
            'id': range(1, 1001),
            'name': [f'Item_{i}' for i in range(1, 1001)],
            'value': np.random.rand(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'date': pd.date_range('2023-01-01', periods=1000)
        })
        
    def tearDown(self):
        """Clean up test environment."""
        import time
        import os
        
        # Try to remove the temp directory with retries to handle file locks
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(self.temp_dir)
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Small delay to allow file handles to be released
                    continue
                else:
                    # On final attempt, try to remove individual files that might be locked
                    try:
                        for file_path in self.temp_dir.rglob('*'):
                            if file_path.is_file():
                                try:
                                    file_path.unlink()
                                except PermissionError:
                                    pass  # Ignore individual file errors
                        self.temp_dir.rmdir()
                    except Exception:
                        pass  # Final fallback - just ignore cleanup errors
    
    def _create_test_csv(self, filename: str = 'test.csv', df=None) -> Path:
        """Create a test CSV file."""
        file_path = self.temp_dir / filename
        test_df = df if df is not None else self.sample_df
        test_df.to_csv(file_path, index=False)
        return file_path
    
    def _create_test_excel(self, filename: str = 'test.xlsx') -> Path:
        """Create a test Excel file."""
        file_path = self.temp_dir / filename
        self.sample_df.to_excel(file_path, index=False)
        return file_path
    
    def _create_test_json(self, filename: str = 'test.json') -> Path:
        """Create a test JSON file."""
        file_path = self.temp_dir / filename
        self.sample_df.to_json(file_path, orient='records')
        return file_path
    
    def test_file_validation_success(self):
        """Test successful file validation."""
        csv_file = self._create_test_csv()
        
        # Should not raise any exception
        try:
            self.file_handler._validate_file(csv_file)
        except Exception as e:
            self.fail(f"File validation failed unexpectedly: {e}")
    
    def test_file_validation_not_found(self):
        """Test file validation with non-existent file."""
        non_existent = self.temp_dir / 'does_not_exist.csv'
        
        with self.assertRaises(FileHandlingError):
            self.file_handler._validate_file(non_existent)
    
    def test_file_validation_unsupported_format(self):
        """Test file validation with unsupported format."""
        unsupported_file = self.temp_dir / 'test.txt'
        unsupported_file.write_text('some text')
        
        with self.assertRaises(UnsupportedFileFormatError):
            self.file_handler._validate_file(unsupported_file)
    
    @patch('src.file_utils.MAX_FILE_SIZE', 100)  # 100 bytes limit for testing
    def test_file_validation_too_large(self):
        """Test file validation with oversized file."""
        large_df = pd.DataFrame({'col': ['x' * 50] * 100})  # Should exceed 100 bytes
        large_file = self._create_test_csv('large.csv', large_df)
        
        with self.assertRaises(FileSizeError):
            self.file_handler._validate_file(large_file)
    
    def test_get_file_info_csv(self):
        """Test getting file info for CSV file."""
        csv_file = self._create_test_csv()
        info = self.file_handler.get_file_info(csv_file)
        
        self.assertEqual(info['name'], 'test.csv')
        self.assertEqual(info['extension'], '.csv')
        self.assertGreater(info['size_bytes'], 0)
        self.assertGreater(info['estimated_rows'], 0)
        self.assertIn('path', info)
        self.assertIn('size_mb', info)
    
    def test_get_file_info_excel(self):
        """Test getting file info for Excel file."""
        excel_file = self._create_test_excel()
        info = self.file_handler.get_file_info(excel_file)
        
        self.assertEqual(info['name'], 'test.xlsx')
        self.assertEqual(info['extension'], '.xlsx')
        self.assertIn('sheets', info)
        self.assertIsInstance(info['sheets'], list)
    
    def test_get_file_info_json(self):
        """Test getting file info for JSON file."""
        json_file = self._create_test_json()
        info = self.file_handler.get_file_info(json_file)
        
        self.assertEqual(info['name'], 'test.json')
        self.assertEqual(info['extension'], '.json')
        self.assertGreater(info['size_bytes'], 0)
    
    def test_read_csv_file_complete(self):
        """Test reading complete CSV file."""
        csv_file = self._create_test_csv()
        df = self.file_handler.read_file(csv_file, use_chunks=False)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        self.assertListEqual(list(df.columns), list(self.sample_df.columns))
    
    def test_read_csv_file_chunks(self):
        """Test reading CSV file in chunks."""
        csv_file = self._create_test_csv()
        chunk_iterator = self.file_handler.read_file(
            csv_file, use_chunks=True, chunk_size=100
        )
        
        chunks = list(chunk_iterator)
        total_rows = sum(len(chunk) for chunk in chunks)
        
        self.assertEqual(total_rows, 1000)
        self.assertEqual(len(chunks), 10)  # 1000 rows / 100 chunk_size
    
    def test_read_excel_file(self):
        """Test reading Excel file."""
        excel_file = self._create_test_excel()
        df = self.file_handler.read_file(excel_file)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        # Note: Excel might have slight type differences
        self.assertEqual(len(df.columns), len(self.sample_df.columns))
    
    def test_read_json_file(self):
        """Test reading JSON file."""
        json_file = self._create_test_json()
        df = self.file_handler.read_file(json_file)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        self.assertEqual(len(df.columns), len(self.sample_df.columns))
    
    def test_read_corrupted_file(self):
        """Test reading corrupted file."""
        corrupted_file = self.temp_dir / 'corrupted.csv'
        corrupted_file.write_text('invalid,csv,content\n1,2\nno,matching,columns,here')
        
        # Should handle gracefully but might still succeed with warnings
        try:
            df = self.file_handler.read_file(corrupted_file)
            # If it succeeds, check it's a DataFrame
            self.assertIsInstance(df, pd.DataFrame)
        except FileCorruptionError:
            # This is also acceptable
            pass
    
    def test_process_large_file_in_chunks(self):
        """Test processing file in chunks with custom processor."""
        csv_file = self._create_test_csv()
        
        def test_processor(chunk, chunk_num):
            return {
                'chunk': chunk_num,
                'rows': len(chunk),
                'mean_value': chunk['value'].mean()
            }
        
        results = self.file_handler.process_large_file_in_chunks(
            csv_file, test_processor, chunk_size=200, show_progress=False
        )
        
        self.assertEqual(len(results), 5)  # 1000 rows / 200 chunk_size
        total_rows = sum(r['rows'] for r in results)
        self.assertEqual(total_rows, 1000)
        
        # Check chunk numbers are sequential
        chunk_numbers = [r['chunk'] for r in results]
        self.assertEqual(chunk_numbers, list(range(5)))
    
    def test_save_dataframe_csv(self):
        """Test saving DataFrame as CSV."""
        output_file = self.temp_dir / 'output.csv'
        
        self.file_handler.save_dataframe(self.sample_df, output_file)
        
        self.assertTrue(output_file.exists())
        
        # Read back and verify
        saved_df = pd.read_csv(output_file)
        self.assertEqual(len(saved_df), len(self.sample_df))
        self.assertListEqual(list(saved_df.columns), list(self.sample_df.columns))
    
    def test_save_dataframe_excel(self):
        """Test saving DataFrame as Excel."""
        output_file = self.temp_dir / 'output.xlsx'
        
        self.file_handler.save_dataframe(self.sample_df, output_file)
        
        self.assertTrue(output_file.exists())
        
        # Read back and verify
        saved_df = pd.read_excel(output_file)
        self.assertEqual(len(saved_df), len(self.sample_df))
    
    def test_save_dataframe_json(self):
        """Test saving DataFrame as JSON."""
        output_file = self.temp_dir / 'output.json'
        
        self.file_handler.save_dataframe(self.sample_df, output_file)
        
        self.assertTrue(output_file.exists())
        
        # Read back and verify
        saved_df = pd.read_json(output_file)
        self.assertEqual(len(saved_df), len(self.sample_df))
    
    def test_save_dataframe_unsupported_format(self):
        """Test saving DataFrame with unsupported format."""
        output_file = self.temp_dir / 'output.txt'
        
        with self.assertRaises(FileHandlingError):
            self.file_handler.save_dataframe(self.sample_df, output_file)
    
    @patch('psutil.virtual_memory')
    def test_memory_usage_check_normal(self, mock_memory):
        """Test memory usage check under normal conditions."""
        mock_memory.return_value.percent = 50.0  # 50% usage
        
        # Should not raise exception
        try:
            self.file_handler._check_memory_usage()
        except InsufficientMemoryError:
            self.fail("Memory check failed unexpectedly")
    
    @patch('psutil.virtual_memory')
    def test_memory_usage_check_high(self, mock_memory):
        """Test memory usage check with high memory usage."""
        mock_memory.return_value.percent = 90.0  # 90% usage (above 80% threshold)
        
        with self.assertRaises(InsufficientMemoryError):
            self.file_handler._check_memory_usage()
    
    def test_estimate_csv_rows(self):
        """Test CSV row estimation."""
        csv_file = self._create_test_csv()
        estimated_rows = self.file_handler._estimate_csv_rows(csv_file)
        
        # Should be close to actual (1000 rows)
        self.assertGreater(estimated_rows, 800)  # Allow some variance
        self.assertLess(estimated_rows, 1200)
    
    def test_get_excel_sheets(self):
        """Test getting Excel sheet names."""
        excel_file = self._create_test_excel()
        sheets = self.file_handler._get_excel_sheets(excel_file)
        
        self.assertIsInstance(sheets, list)
        self.assertGreater(len(sheets), 0)
        self.assertIn('Sheet1', sheets)  # Default sheet name


class TestSampleProcessor(unittest.TestCase):
    """Test cases for sample processor function."""
    
    def test_create_sample_processor(self):
        """Test sample processor creation and execution."""
        processor = create_sample_processor()
        
        # Create test data
        test_df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', 'b', 'c', None, 'e']
        })
        
        result = processor(test_df, 0)
        
        self.assertIsInstance(result, dict)
        self.assertIn('chunk_number', result)
        self.assertIn('rows', result)
        self.assertIn('columns', result)
        self.assertIn('memory_usage_mb', result)
        self.assertIn('null_values', result)
        
        self.assertEqual(result['chunk_number'], 0)
        self.assertEqual(result['rows'], 5)
        self.assertEqual(result['columns'], 2)
        self.assertEqual(result['null_values'], 2)  # One None in each column


class TestConfigAndSetup(unittest.TestCase):
    """Test configuration and setup functions."""
    
    def test_ensure_directories(self):
        """Test directory creation."""
        from src.config import ensure_directories
        
        # Should not raise any exceptions
        try:
            ensure_directories()
        except Exception as e:
            self.fail(f"Directory creation failed: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
