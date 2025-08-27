"""
Comprehensive data profiling module for Smart Data Cleaning System - Phase 2.
Generates detailed data profiles with quality metrics and structured JSON output.
"""

import json
import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .config import CHUNK_SIZE
from .exceptions import ProfilingError, InsufficientMemoryError
from .file_utils import FileHandler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


class DataProfiler:
    """
    Comprehensive data profiling with quality assessment and structured output.
    """
    
    def __init__(self, memory_threshold: float = 0.8):
        """
        Initialize DataProfiler.
        
        Args:
            memory_threshold: Maximum memory usage threshold
        """
        self.memory_threshold = memory_threshold
        self.logger = self._setup_logger()
        self.file_handler = FileHandler(memory_threshold)
        
        # Pattern definitions for data type detection
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+\d{1,3}[- ]?)?\(?\d{1,4}\)?[- ]?\d{1,4}[- ]?\d{1,9}'),
            'url': re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'),
            'date_iso': re.compile(r'\d{4}-\d{2}-\d{2}'),
            'date_us': re.compile(r'\d{1,2}/\d{1,2}/\d{4}'),
            'date_eu': re.compile(r'\d{1,2}\.\d{1,2}\.\d{4}'),
            'currency': re.compile(r'[\$£€¥]\s?\d+(?:,\d{3})*(?:\.\d{2})?'),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
            'social_security': re.compile(r'\d{3}-\d{2}-\d{4}'),
            'credit_card': re.compile(r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}'),
            'postal_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for profiling operations."""
        logger = logging.getLogger(f"{__name__}.DataProfiler")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        # Prevent duplicate logs by stopping propagation to root logger
        logger.propagate = False
        return logger
    
    def profile_dataset(self, 
                       file_path: Union[str, Path],
                       chunk_size: Optional[int] = None,
                       sample_size: Optional[int] = None,
                       export_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data profile for a dataset.
        
        Args:
            file_path: Path to the dataset file
            chunk_size: Size of chunks for large files
            sample_size: Sample size for very large datasets
            export_path: Path to export JSON profile
            
        Returns:
            Complete data profile dictionary
        """
        file_path = Path(file_path)
        chunk_size = chunk_size or CHUNK_SIZE
        
        self.logger.info(f"Starting comprehensive data profiling: {file_path.name}")
        
        try:
            # Get basic file information
            file_info = self.file_handler.get_file_info(file_path)
            
            # Determine processing strategy
            use_chunks = file_info['size_mb'] > 100 or (
                'estimated_rows' in file_info and file_info['estimated_rows'] > 50000
            )
            
            if use_chunks:
                profile = self._profile_large_dataset(file_path, chunk_size, sample_size)
            else:
                df = self.file_handler.read_file(file_path, use_chunks=False)
                profile = self._profile_complete_dataset(df, file_info)
            
            # Add metadata
            profile['profiling_metadata'] = {
                'profiled_at': datetime.now().isoformat(),
                'profiler_version': '2.0.0',
                'processing_method': 'chunked' if use_chunks else 'complete',
                'sample_size': sample_size if sample_size else 'full'
            }
            
            # Export to JSON if requested
            if export_path:
                self._export_profile(profile, export_path)
            
            self.logger.info(f"Data profiling completed: {file_path.name}")
            return profile
            
        except Exception as e:
            raise ProfilingError(f"Failed to profile dataset: {str(e)}", str(file_path))
    
    def _profile_complete_dataset(self, df: pd.DataFrame, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Profile a complete dataset loaded in memory."""
        self.logger.info(f"Profiling complete dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Dataset-level information
        dataset_profile = {
            'dataset_info': self._generate_dataset_info(df, file_info),
            'column_profiles': {},
            'data_relationships': self._analyze_data_relationships(df),
            'data_quality_summary': {}
        }
        
        # Profile each column
        column_quality_scores = []
        
        with tqdm(total=len(df.columns), desc="Profiling columns") as pbar:
            for column in df.columns:
                try:
                    column_profile = self._profile_column(df[column], column, df)
                    dataset_profile['column_profiles'][column] = column_profile
                    column_quality_scores.append(column_profile['quality_score'])
                except Exception as e:
                    self.logger.warning(f"Failed to profile column '{column}': {e}")
                    dataset_profile['column_profiles'][column] = {
                        'error': str(e),
                        'quality_score': 0.0
                    }
                    column_quality_scores.append(0.0)
                finally:
                    pbar.update(1)
        
        # Calculate overall quality score
        dataset_profile['dataset_info']['overall_quality_score'] = np.mean(column_quality_scores)
        dataset_profile['data_quality_summary'] = self._generate_quality_summary(
            dataset_profile['column_profiles']
        )
        
        return dataset_profile
    
    def _profile_large_dataset(self, 
                              file_path: Path, 
                              chunk_size: int,
                              sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Profile large dataset using chunked processing."""
        self.logger.info(f"Profiling large dataset in chunks of {chunk_size:,} rows")
        
        # Initialize accumulators
        file_info = self.file_handler.get_file_info(file_path)
        chunk_profiles = []
        column_stats = {}
        total_rows = 0
        
        # Read and process chunks
        chunk_iterator = self.file_handler.read_file(
            file_path, use_chunks=True, chunk_size=chunk_size
        )
        
        chunk_count = 0
        sample_collected = 0 if sample_size else float('inf')
        
        for chunk_num, chunk in enumerate(chunk_iterator):
            if sample_size and sample_collected >= sample_size:
                break
                
            try:
                self.file_handler._check_memory_usage()
                
                # Sample from chunk if needed
                if sample_size:
                    remaining_sample = sample_size - sample_collected
                    if remaining_sample < len(chunk):
                        chunk = chunk.sample(n=remaining_sample, random_state=42)
                    sample_collected += len(chunk)
                
                # Process chunk
                chunk_profile = self._process_chunk_for_profiling(chunk, chunk_num)
                chunk_profiles.append(chunk_profile)
                total_rows += len(chunk)
                chunk_count += 1
                
                # Update column statistics
                self._update_column_stats(column_stats, chunk, chunk_num == 0)
                
                if chunk_count % 10 == 0:
                    self.logger.info(f"Processed {chunk_count} chunks, {total_rows:,} rows")
                    
            except InsufficientMemoryError:
                self.logger.warning(f"Memory threshold reached at chunk {chunk_num}")
                break
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_num}: {e}")
                continue
        
        # Combine chunk results
        return self._combine_chunk_profiles(chunk_profiles, column_stats, file_info, total_rows)
    
    def _generate_dataset_info(self, df: pd.DataFrame, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dataset-level information."""
        return {
            'filename': file_info['name'],
            'file_path': file_info['path'],
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'file_size_mb': file_info['size_mb'],
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'overall_quality_score': 0.0,  # Will be calculated later
            'column_names': list(df.columns),
            'data_types_summary': self._summarize_data_types(df),
        }
    
    def _summarize_data_types(self, df: pd.DataFrame) -> Dict[str, int]:
        """Summarize data types across the dataset."""
        type_counts = {}
        for dtype in df.dtypes:
            dtype_str = str(dtype)
            if dtype_str.startswith('int'):
                dtype_category = 'integer'
            elif dtype_str.startswith('float'):
                dtype_category = 'float'
            elif dtype_str == 'object':
                dtype_category = 'text'
            elif dtype_str.startswith('datetime'):
                dtype_category = 'datetime'
            elif dtype_str == 'bool':
                dtype_category = 'boolean'
            else:
                dtype_category = 'other'
            
            type_counts[dtype_category] = type_counts.get(dtype_category, 0) + 1
        
        return type_counts
    
    def _profile_column(self, series: pd.Series, column_name: str, full_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive profile for a single column."""
        profile = {
            'column_name': column_name,
            'data_type': self._detect_data_type(series),
            'pandas_dtype': str(series.dtype),
            'basic_stats': self._calculate_basic_stats(series),
            'null_analysis': self._analyze_nulls(series),
            'unique_analysis': self._analyze_unique_values(series),
            'pattern_analysis': self._analyze_patterns(series),
            'outlier_analysis': self._detect_outliers(series),
            'quality_issues': [],
            'quality_score': 0.0,
            'recommended_actions': []
        }
        
        # Add type-specific analysis
        if profile['data_type']['primary_type'] == 'numeric':
            profile['numeric_analysis'] = self._analyze_numeric_column(series)
        elif profile['data_type']['primary_type'] == 'text':
            profile['text_analysis'] = self._analyze_text_column(series)
        elif profile['data_type']['primary_type'] == 'datetime':
            profile['datetime_analysis'] = self._analyze_datetime_column(series)
        
        # Calculate quality score and issues
        profile['quality_score'], profile['quality_issues'], profile['recommended_actions'] = (
            self._assess_column_quality(series, profile)
        )
        
        return profile
    
    def _detect_data_type(self, series: pd.Series) -> Dict[str, Any]:
        """Detect data type with confidence scores."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {
                'primary_type': 'empty',
                'confidence': 1.0,
                'type_consistency': 0.0,
                'mixed_types': []
            }
        
        # Convert to string for pattern analysis
        str_series = non_null_series.astype(str)
        
        type_scores = {
            'numeric': self._score_numeric_type(non_null_series),
            'datetime': self._score_datetime_type(str_series),
            'boolean': self._score_boolean_type(str_series),
            'categorical': self._score_categorical_type(non_null_series),
            'text': self._score_text_type(str_series)
        }
        
        # Find primary type
        primary_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[primary_type]
        
        # Check for mixed types
        mixed_types = [t for t, score in type_scores.items() if score > 0.1 and t != primary_type]
        type_consistency = 1.0 - len(mixed_types) * 0.2
        
        return {
            'primary_type': primary_type,
            'confidence': confidence,
            'type_consistency': max(0.0, type_consistency),
            'mixed_types': mixed_types,
            'all_scores': type_scores
        }
    
    def _score_numeric_type(self, series: pd.Series) -> float:
        """Score how likely the series is numeric."""
        try:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_numeric = numeric_series.notna().sum()
            total_values = len(series)
            
            if total_values == 0:
                return 0.0
            
            return valid_numeric / total_values
        except Exception:
            return 0.0
    
    def _score_datetime_type(self, str_series: pd.Series) -> float:
        """Score how likely the series contains datetime values."""
        if len(str_series) == 0:
            return 0.0
        
        datetime_patterns = 0
        total_values = len(str_series)
        
        for value in str_series.head(min(100, len(str_series))):  # Sample for performance
            if any(pattern.search(str(value)) for pattern in [
                self.patterns['date_iso'], 
                self.patterns['date_us'], 
                self.patterns['date_eu']
            ]):
                datetime_patterns += 1
        
        # Try pandas datetime conversion (suppress noisy inference warnings)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                pd.to_datetime(str_series.head(min(100, len(str_series))), errors='coerce')
                datetime_convertible = pd.to_datetime(str_series, errors='coerce').notna().sum()
            return datetime_convertible / total_values
        except Exception:
            return datetime_patterns / min(100, total_values)
    
    def _score_boolean_type(self, str_series: pd.Series) -> float:
        """Score how likely the series contains boolean values."""
        if len(str_series) == 0:
            return 0.0
        
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 't', 'f'}
        unique_lower = set(str_series.str.lower().unique())
        
        if unique_lower.issubset(boolean_values):
            return 1.0
        elif len(unique_lower.intersection(boolean_values)) > 0:
            return len(unique_lower.intersection(boolean_values)) / len(unique_lower)
        else:
            return 0.0
    
    def _score_categorical_type(self, series: pd.Series) -> float:
        """Score how likely the series is categorical."""
        total_values = len(series)
        unique_values = series.nunique()
        
        if total_values == 0:
            return 0.0
        
        # Categorical if unique values are less than 20% of total or less than 50
        uniqueness_ratio = unique_values / total_values
        
        if unique_values <= 50 and uniqueness_ratio <= 0.2:
            return 1.0 - uniqueness_ratio
        else:
            return 0.0
    
    def _score_text_type(self, str_series: pd.Series) -> float:
        """Score how likely the series contains free text."""
        if len(str_series) == 0:
            return 0.0
        
        # Check average length and character diversity
        avg_length = str_series.str.len().mean()
        unique_ratio = str_series.nunique() / len(str_series)
        
        # Text typically has longer strings and high uniqueness
        length_score = min(1.0, avg_length / 50)  # Normalize by 50 chars
        uniqueness_score = min(1.0, unique_ratio * 2)  # Boost uniqueness
        
        return (length_score + uniqueness_score) / 2
    
    def _calculate_basic_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistics for the series."""
        stats = {
            'count': len(series),
            'non_null_count': series.notna().sum(),
            'null_count': series.isna().sum(),
            'null_percentage': (series.isna().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0,
        }
        
        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series.dropna()
            if len(numeric_series) > 0:
                stats.update({
                    'mean': float(numeric_series.mean()),
                    'median': float(numeric_series.median()),
                    'std': float(numeric_series.std()),
                    'min': float(numeric_series.min()),
                    'max': float(numeric_series.max()),
                    'q25': float(numeric_series.quantile(0.25)),
                    'q75': float(numeric_series.quantile(0.75)),
                    'skewness': float(numeric_series.skew()),
                    'kurtosis': float(numeric_series.kurtosis())
                })
        
        return stats
    
    def _analyze_nulls(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze null value patterns."""
        null_analysis = {
            'null_count': series.isna().sum(),
            'null_percentage': (series.isna().sum() / len(series)) * 100,
            'consecutive_nulls': self._find_consecutive_nulls(series),
            'null_pattern': self._analyze_null_pattern(series)
        }
        
        return null_analysis
    
    def _find_consecutive_nulls(self, series: pd.Series) -> Dict[str, Any]:
        """Find consecutive null sequences."""
        null_mask = series.isna()
        
        if not null_mask.any():
            return {'max_consecutive': 0, 'sequences': []}
        
        # Find consecutive null sequences
        sequences = []
        current_seq = 0
        max_consecutive = 0
        
        for is_null in null_mask:
            if is_null:
                current_seq += 1
            else:
                if current_seq > 0:
                    sequences.append(current_seq)
                    max_consecutive = max(max_consecutive, current_seq)
                current_seq = 0
        
        # Don't forget the last sequence
        if current_seq > 0:
            sequences.append(current_seq)
            max_consecutive = max(max_consecutive, current_seq)
        
        return {
            'max_consecutive': max_consecutive,
            'sequences': sequences,
            'sequence_count': len(sequences)
        }
    
    def _analyze_null_pattern(self, series: pd.Series) -> str:
        """Analyze the pattern of null values."""
        null_percentage = (series.isna().sum() / len(series)) * 100
        
        if null_percentage == 0:
            return 'no_nulls'
        elif null_percentage < 5:
            return 'sparse_nulls'
        elif null_percentage < 20:
            return 'moderate_nulls'
        elif null_percentage < 50:
            return 'heavy_nulls'
        else:
            return 'mostly_null'
    
    def _analyze_unique_values(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze unique value distribution."""
        value_counts = series.value_counts()
        
        analysis = {
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0,
            'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'least_frequent_value': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
        }
        
        # Add top values (limit to avoid huge dictionaries)
        top_values = value_counts.head(10).to_dict()
        analysis['top_values'] = {str(k): int(v) for k, v in top_values.items()}
        
        return analysis
    
    def _analyze_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in string data."""
        str_series = series.astype(str).dropna()
        
        if len(str_series) == 0:
            return {'patterns_found': {}}
        
        patterns_found = {}
        sample_size = min(1000, len(str_series))  # Limit for performance
        sample = str_series.sample(n=sample_size, random_state=42)
        
        for pattern_name, pattern in self.patterns.items():
            matches = sum(1 for value in sample if pattern.search(str(value)))
            if matches > 0:
                patterns_found[pattern_name] = {
                    'match_count': matches,
                    'match_percentage': (matches / sample_size) * 100
                }
        
        return {'patterns_found': patterns_found}
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        if not pd.api.types.is_numeric_dtype(series):
            return {'method': 'not_applicable', 'outlier_count': 0}
        
        numeric_series = series.dropna()
        if len(numeric_series) < 3:
            return {'method': 'insufficient_data', 'outlier_count': 0}
        
        outlier_analysis = {}
        
        # IQR method
        try:
            q25, q75 = numeric_series.quantile([0.25, 0.75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            iqr_outliers = ((numeric_series < lower_bound) | (numeric_series > upper_bound)).sum()
            outlier_analysis['iqr_method'] = {
                'outlier_count': int(iqr_outliers),
                'outlier_percentage': (iqr_outliers / len(numeric_series)) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        except Exception as e:
            outlier_analysis['iqr_method'] = {'error': str(e)}
        
        # Z-score method
        try:
            z_scores = np.abs(stats.zscore(numeric_series))
            z_outliers = (z_scores > 3).sum()
            outlier_analysis['zscore_method'] = {
                'outlier_count': int(z_outliers),
                'outlier_percentage': (z_outliers / len(numeric_series)) * 100,
                'threshold': 3.0
            }
        except Exception as e:
            outlier_analysis['zscore_method'] = {'error': str(e)}
        
        return outlier_analysis
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Perform detailed numeric analysis."""
        # Coerce numeric-like values (e.g., '15.6', '1,234', ' 42 ')
        numeric_series = pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return {'error': 'no_numeric_data'}
        
        analysis = {
            'distribution_analysis': self._analyze_distribution(numeric_series),
            'range_analysis': self._analyze_numeric_range(numeric_series),
            'precision_analysis': self._analyze_numeric_precision(numeric_series)
        }
        
        return analysis
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of numeric data."""
        try:
            # Basic distribution metrics
            skewness = float(series.skew())
            kurtosis = float(series.kurtosis())
            
            # Classify distribution shape
            if abs(skewness) < 0.5:
                skew_interpretation = 'approximately_symmetric'
            elif skewness > 0.5:
                skew_interpretation = 'right_skewed'
            else:
                skew_interpretation = 'left_skewed'
            
            if abs(kurtosis) < 0.5:
                kurtosis_interpretation = 'mesokurtic'
            elif kurtosis > 0.5:
                kurtosis_interpretation = 'leptokurtic'
            else:
                kurtosis_interpretation = 'platykurtic'
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'skew_interpretation': skew_interpretation,
                'kurtosis_interpretation': kurtosis_interpretation
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_numeric_range(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the range and spread of numeric data."""
        return {
            'range': float(series.max() - series.min()),
            'interquartile_range': float(series.quantile(0.75) - series.quantile(0.25)),
            'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else float('inf'),
            'zero_count': int((series == 0).sum()),
            'negative_count': int((series < 0).sum()),
            'positive_count': int((series > 0).sum())
        }
    
    def _analyze_numeric_precision(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric precision and decimal places."""
        try:
            # Convert to string to analyze decimal places
            str_series = series.astype(str)
            
            # Count decimal places
            decimal_places = []
            integer_count = 0
            
            for value in str_series:
                if '.' in value:
                    decimal_count = len(value.split('.')[1])
                    decimal_places.append(decimal_count)
                else:
                    integer_count += 1
                    decimal_places.append(0)
            
            return {
                'integer_count': integer_count,
                'decimal_count': len(series) - integer_count,
                'max_decimal_places': max(decimal_places) if decimal_places else 0,
                'avg_decimal_places': np.mean(decimal_places) if decimal_places else 0,
                'is_integer_only': integer_count == len(series)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Perform detailed text analysis."""
        str_series = series.astype(str).dropna()
        
        if len(str_series) == 0:
            return {'error': 'no_text_data'}
        
        analysis = {
            'length_analysis': self._analyze_text_length(str_series),
            'character_analysis': self._analyze_text_characters(str_series),
            'language_analysis': self._analyze_text_language(str_series)
        }
        
        return analysis
    
    def _analyze_text_length(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text length patterns."""
        lengths = series.str.len()
        
        return {
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'avg_length': float(lengths.mean()),
            'std_length': float(lengths.std()),
            'empty_strings': int((lengths == 0).sum()),
            'single_character': int((lengths == 1).sum())
        }
    
    def _analyze_text_characters(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze character patterns in text."""
        sample = series.head(min(100, len(series)))  # Sample for performance
        
        has_digits = sum(1 for text in sample if any(c.isdigit() for c in str(text)))
        has_letters = sum(1 for text in sample if any(c.isalpha() for c in str(text)))
        has_spaces = sum(1 for text in sample if ' ' in str(text))
        has_special_chars = sum(1 for text in sample if any(not c.isalnum() and c != ' ' for c in str(text)))
        
        sample_size = len(sample)
        
        return {
            'contains_digits_pct': (has_digits / sample_size) * 100,
            'contains_letters_pct': (has_letters / sample_size) * 100,
            'contains_spaces_pct': (has_spaces / sample_size) * 100,
            'contains_special_chars_pct': (has_special_chars / sample_size) * 100
        }
    
    def _analyze_text_language(self, series: pd.Series) -> Dict[str, Any]:
        """Basic language analysis of text."""
        sample = series.head(min(50, len(series)))
        
        # Simple heuristics for language detection
        total_chars = 0
        ascii_chars = 0
        
        for text in sample:
            text_str = str(text)
            total_chars += len(text_str)
            ascii_chars += sum(1 for c in text_str if ord(c) < 128)
        
        ascii_percentage = (ascii_chars / total_chars * 100) if total_chars > 0 else 0
        
        return {
            'ascii_percentage': ascii_percentage,
            'likely_encoding': 'ascii' if ascii_percentage > 95 else 'unicode',
            'sample_analyzed': len(sample)
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Perform detailed datetime analysis."""
        # Try to convert to datetime
        try:
            dt_series = pd.to_datetime(series, errors='coerce').dropna()
            
            if len(dt_series) == 0:
                return {'error': 'no_valid_datetimes'}
            
            analysis = {
                'date_range': {
                    'min_date': dt_series.min().isoformat(),
                    'max_date': dt_series.max().isoformat(),
                    'date_span_days': (dt_series.max() - dt_series.min()).days
                },
                'temporal_patterns': self._analyze_temporal_patterns(dt_series),
                'format_consistency': (len(dt_series) / len(series)) * 100
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_temporal_patterns(self, dt_series: pd.Series) -> Dict[str, Any]:
        """Analyze temporal patterns in datetime data."""
        try:
            # Extract time components
            years = dt_series.dt.year.value_counts()
            months = dt_series.dt.month.value_counts()
            weekdays = dt_series.dt.dayofweek.value_counts()
            
            return {
                'year_distribution': years.head(10).to_dict(),
                'month_distribution': months.to_dict(),
                'weekday_distribution': weekdays.to_dict(),
                'has_time_component': any(dt_series.dt.time != pd.Timestamp('00:00:00').time())
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_column_quality(self, series: pd.Series, profile: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess column data quality and generate recommendations."""
        quality_score = 1.0
        issues = []
        recommendations = []
        
        # Null value assessment
        null_pct = profile['basic_stats']['null_percentage']
        if null_pct > 50:
            quality_score -= 0.4
            issues.append(f"High null percentage ({null_pct:.1f}%)")
            recommendations.append("Consider dropping column or imputing missing values")
        elif null_pct > 20:
            quality_score -= 0.2
            issues.append(f"Moderate null percentage ({null_pct:.1f}%)")
            recommendations.append("Consider imputing missing values")
        elif null_pct > 5:
            quality_score -= 0.1
            issues.append(f"Some null values present ({null_pct:.1f}%)")
            recommendations.append("Review null value patterns")
        
        # Type consistency assessment
        type_confidence = profile['data_type']['confidence']
        if type_confidence < 0.8:
            quality_score -= 0.3
            issues.append(f"Low type consistency (confidence: {type_confidence:.2f})")
            recommendations.append("Consider data type conversion or cleaning")
        
        # Uniqueness assessment
        unique_pct = profile['basic_stats']['unique_percentage']
        if unique_pct < 1 and profile['data_type']['primary_type'] != 'categorical':
            issues.append(f"Low uniqueness ({unique_pct:.1f}%)")
            if unique_pct < 50:
                quality_score -= 0.1
                recommendations.append("Check for data entry errors or consider categorizing")
        
        # Outlier assessment
        if 'outlier_analysis' in profile and 'iqr_method' in profile['outlier_analysis']:
            iqr_outliers = profile['outlier_analysis']['iqr_method'].get('outlier_percentage', 0)
            if iqr_outliers > 10:
                quality_score -= 0.2
                issues.append(f"High outlier percentage ({iqr_outliers:.1f}%)")
                recommendations.append("Review outliers for data entry errors")
            elif iqr_outliers > 5:
                quality_score -= 0.1
                issues.append(f"Some outliers detected ({iqr_outliers:.1f}%)")
                recommendations.append("Investigate outlier patterns")
        
        # Mixed types assessment
        if profile['data_type']['mixed_types']:
            quality_score -= 0.2
            issues.append(f"Mixed data types detected: {profile['data_type']['mixed_types']}")
            recommendations.append("Standardize data format")
        
        # Ensure quality score is between 0 and 1
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score, issues, recommendations
    
    def _analyze_data_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between columns."""
        relationships = {
            'correlation_analysis': {},
            'potential_keys': [],
            'duplicate_columns': []
        }
        
        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                
                # Find high correlations
                high_correlations = []
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:  # Avoid duplicates
                            corr_value = corr_matrix.loc[col1, col2]
                            if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                                high_correlations.append({
                                    'column1': col1,
                                    'column2': col2,
                                    'correlation': float(corr_value)
                                })
                
                relationships['correlation_analysis'] = {
                    'high_correlations': high_correlations,
                    'numeric_columns_analyzed': len(numeric_cols)
                }
            except Exception as e:
                relationships['correlation_analysis'] = {'error': str(e)}
        
        # Identify potential keys (high uniqueness)
        for col in df.columns:
            uniqueness = df[col].nunique() / len(df)
            if uniqueness > 0.95:
                relationships['potential_keys'].append({
                    'column': col,
                    'uniqueness': uniqueness,
                    'unique_count': df[col].nunique()
                })
        
        # Find duplicate columns (same values)
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:
                    try:
                        if df[col1].equals(df[col2]):
                            relationships['duplicate_columns'].append([col1, col2])
                    except Exception:
                        continue
        
        return relationships
    
    def _generate_quality_summary(self, column_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall data quality summary."""
        quality_scores = [
            profile.get('quality_score', 0.0) 
            for profile in column_profiles.values() 
            if isinstance(profile, dict) and 'quality_score' in profile
        ]
        
        if not quality_scores:
            return {'error': 'no_quality_scores_available'}
        
        # Quality distribution
        excellent = sum(1 for score in quality_scores if score >= 0.9)
        good = sum(1 for score in quality_scores if 0.7 <= score < 0.9)
        fair = sum(1 for score in quality_scores if 0.5 <= score < 0.7)
        poor = sum(1 for score in quality_scores if score < 0.5)
        
        # Common issues
        all_issues = []
        for profile in column_profiles.values():
            if isinstance(profile, dict) and 'quality_issues' in profile:
                all_issues.extend(profile['quality_issues'])
        
        issue_counts = {}
        for issue in all_issues:
            # Extract issue type (before percentage if present)
            issue_type = issue.split('(')[0].strip()
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        return {
            'overall_score': np.mean(quality_scores),
            'quality_distribution': {
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor
            },
            'total_columns': len(quality_scores),
            'common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def _process_chunk_for_profiling(self, chunk: pd.DataFrame, chunk_num: int) -> Dict[str, Any]:
        """Process a single chunk for profiling."""
        return {
            'chunk_number': chunk_num,
            'row_count': len(chunk),
            'column_count': len(chunk.columns),
            'memory_usage_mb': chunk.memory_usage(deep=True).sum() / (1024 ** 2),
            'null_counts': chunk.isnull().sum().to_dict(),
            'basic_stats': {}  # Would contain basic stats per column
        }
    
    def _update_column_stats(self, column_stats: Dict, chunk: pd.DataFrame, is_first_chunk: bool):
        """Update running column statistics with chunk data."""
        if is_first_chunk:
            # Initialize column stats
            for col in chunk.columns:
                column_stats[col] = {
                    'total_count': 0,
                    'null_count': 0,
                    'unique_values': set(),
                    'numeric_sum': 0.0,
                    'numeric_sum_squares': 0.0,
                    'min_val': None,
                    'max_val': None
                }
        
        # Update stats with chunk data
        for col in chunk.columns:
            if col not in column_stats:
                continue
                
            series = chunk[col].dropna()
            column_stats[col]['total_count'] += len(chunk[col])
            column_stats[col]['null_count'] += chunk[col].isnull().sum()
            
            # Update unique values (limit to avoid memory issues)
            if len(column_stats[col]['unique_values']) < 1000:
                column_stats[col]['unique_values'].update(series.unique())
            
            # Update numeric stats if applicable
            if pd.api.types.is_numeric_dtype(series) and len(series) > 0:
                column_stats[col]['numeric_sum'] += series.sum()
                column_stats[col]['numeric_sum_squares'] += (series ** 2).sum()
                
                current_min = series.min()
                current_max = series.max()
                
                if column_stats[col]['min_val'] is None:
                    column_stats[col]['min_val'] = current_min
                    column_stats[col]['max_val'] = current_max
                else:
                    column_stats[col]['min_val'] = min(column_stats[col]['min_val'], current_min)
                    column_stats[col]['max_val'] = max(column_stats[col]['max_val'], current_max)
    
    def _combine_chunk_profiles(self, 
                              chunk_profiles: List[Dict], 
                              column_stats: Dict, 
                              file_info: Dict, 
                              total_rows: int) -> Dict[str, Any]:
        """Combine chunk profiles into final dataset profile."""
        
        self.logger.info(f"Combining profiles from {len(chunk_profiles)} chunks")
        
        # Create dataset info
        dataset_info = {
            'filename': file_info['name'],
            'file_path': file_info['path'],
            'total_rows': total_rows,
            'total_columns': len(column_stats),
            'file_size_mb': file_info['size_mb'],
            'chunks_processed': len(chunk_profiles),
            'overall_quality_score': 0.0  # Will be calculated
        }
        
        # Create column profiles from aggregated stats
        column_profiles = {}
        quality_scores = []
        
        for col, col_stats in column_stats.items():
            try:
                # Calculate basic metrics
                null_pct = (col_stats['null_count'] / col_stats['total_count']) * 100 if col_stats['total_count'] > 0 else 0
                unique_count = len(col_stats['unique_values'])
                
                profile = {
                    'column_name': col,
                    'basic_stats': {
                        'count': col_stats['total_count'],
                        'null_count': col_stats['null_count'],
                        'null_percentage': null_pct,
                        'unique_count': unique_count,
                        'unique_percentage': (unique_count / col_stats['total_count']) * 100 if col_stats['total_count'] > 0 else 0
                    },
                    'data_type': {'primary_type': 'unknown', 'confidence': 0.0},  # Simplified for chunked
                    'quality_issues': [],
                    'recommended_actions': []
                }
                
                # Add numeric stats if available
                if col_stats['min_val'] is not None:
                    non_null_count = col_stats['total_count'] - col_stats['null_count']
                    if non_null_count > 0:
                        mean = col_stats['numeric_sum'] / non_null_count
                        profile['basic_stats'].update({
                            'mean': mean,
                            'min': col_stats['min_val'],
                            'max': col_stats['max_val']
                        })
                
                # Simple quality assessment
                quality_score = 1.0
                if null_pct > 50:
                    quality_score -= 0.4
                    profile['quality_issues'].append(f"High null percentage ({null_pct:.1f}%)")
                elif null_pct > 20:
                    quality_score -= 0.2
                    profile['quality_issues'].append(f"Moderate null percentage ({null_pct:.1f}%)")
                
                profile['quality_score'] = max(0.0, quality_score)
                quality_scores.append(profile['quality_score'])
                column_profiles[col] = profile
                
            except Exception as e:
                self.logger.warning(f"Error processing column '{col}': {e}")
                column_profiles[col] = {'error': str(e), 'quality_score': 0.0}
                quality_scores.append(0.0)
        
        # Calculate overall quality score
        dataset_info['overall_quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
        
        # Create final profile
        final_profile = {
            'dataset_info': dataset_info,
            'column_profiles': column_profiles,
            'data_quality_summary': self._generate_quality_summary(column_profiles),
            'processing_summary': {
                'chunks_processed': len(chunk_profiles),
                'total_memory_used_mb': sum(cp.get('memory_usage_mb', 0) for cp in chunk_profiles)
            }
        }
        
        return final_profile
    
    def _export_profile(self, profile: Dict[str, Any], export_path: Union[str, Path]) -> None:
        """Export profile to JSON file."""
        export_path = Path(export_path)
        
        # Ensure output directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert profile
        json_ready_profile = convert_numpy_types(profile)
        
        # Export to JSON
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(json_ready_profile, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Data profile exported to: {export_path}")
    
    def generate_profile_summary(self, profile: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the profile."""
        dataset_info = profile.get('dataset_info', {})
        quality_summary = profile.get('data_quality_summary', {})
        
        summary = f"""
📊 DATA PROFILE SUMMARY
=======================

📁 Dataset: {dataset_info.get('filename', 'Unknown')}
📏 Size: {dataset_info.get('total_rows', 0):,} rows × {dataset_info.get('total_columns', 0)} columns
💾 File Size: {dataset_info.get('file_size_mb', 0):.2f} MB
🎯 Overall Quality Score: {dataset_info.get('overall_quality_score', 0):.2f}/1.00

📈 QUALITY DISTRIBUTION:
"""
        
        if 'quality_distribution' in quality_summary:
            qd = quality_summary['quality_distribution']
            summary += (
                f"  ✅ Excellent (≥0.9): {qd.get('excellent', 0)} columns\n"
                f"  ✔️  Good (0.7-0.9): {qd.get('good', 0)} columns\n"
                f"  ⚠️  Fair (0.5-0.7): {qd.get('fair', 0)} columns\n"
                f"  ❌ Poor (<0.5): {qd.get('poor', 0)} columns\n"
            )
        
        if 'common_issues' in quality_summary:
            summary += f"\n🚨 COMMON ISSUES:\n"
            for issue, count in list(quality_summary['common_issues'].items())[:5]:
                summary += f"  • {issue}: {count} columns\n"
        
        return summary