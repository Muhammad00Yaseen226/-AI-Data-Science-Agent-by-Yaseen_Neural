"""
Data Cleaning Module
Handles missing values, duplicates, wrong types, and outliers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class DataCleaner:
    """Cleans data by handling missing values, duplicates, type issues, and outliers."""
    
    def __init__(self):
        self.cleaning_report = {}
    
    def detect_wrong_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect columns with potentially wrong data types."""
        wrong_types = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                if numeric_values.notna().sum() > df[col].notna().sum() * 0.8:
                    wrong_types[col] = 'numeric'
                # Try to convert to datetime
                datetime_values = pd.to_datetime(df[col], errors='coerce')
                if datetime_values.notna().sum() > df[col].notna().sum() * 0.8:
                    wrong_types[col] = 'datetime'
        
        return wrong_types
    
    def fix_wrong_types(self, df: pd.DataFrame, wrong_types: Dict[str, str] = None) -> pd.DataFrame:
        """Fix columns with wrong data types."""
        df_cleaned = df.copy()
        
        if wrong_types is None:
            wrong_types = self.detect_wrong_types(df_cleaned)
        
        type_fixes = {}
        
        for col, target_type in wrong_types.items():
            try:
                if target_type == 'numeric':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    type_fixes[col] = 'numeric'
                elif target_type == 'datetime':
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    type_fixes[col] = 'datetime'
            except Exception as e:
                type_fixes[col] = f'Error: {str(e)}'
        
        self.cleaning_report['type_fixes'] = type_fixes
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto', 
                             numeric_strategy: str = 'mean', 
                             categorical_strategy: str = 'mode',
                             threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values.
        
        Parameters:
        - strategy: 'auto', 'drop', 'fill', or 'threshold'
        - numeric_strategy: 'mean', 'median', 'mode', or 'forward_fill'
        - categorical_strategy: 'mode', 'forward_fill', or 'unknown'
        - threshold: Drop columns/rows with more than this fraction of missing values
        """
        df_cleaned = df.copy()
        missing_info = {}
        
        # Calculate missing values
        missing_counts = df_cleaned.isnull().sum()
        missing_percentages = (missing_counts / len(df_cleaned)) * 100
        
        missing_info['before'] = {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': int((missing_counts > 0).sum()),
            'missing_percentages': missing_percentages.to_dict()
        }
        
        if strategy == 'auto':
            # Drop columns with more than threshold missing values
            cols_to_drop = missing_percentages[missing_percentages > threshold * 100].index.tolist()
            if cols_to_drop:
                df_cleaned = df_cleaned.drop(columns=cols_to_drop)
                missing_info['dropped_columns'] = cols_to_drop
            
            # Fill remaining missing values
            strategy = 'fill'
        
        if strategy == 'threshold':
            # Drop columns with high missing values
            cols_to_drop = missing_percentages[missing_percentages > threshold * 100].index.tolist()
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)
            missing_info['dropped_columns'] = cols_to_drop
        
        if strategy == 'drop':
            # Drop rows with any missing values
            rows_before = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            missing_info['dropped_rows'] = rows_before - len(df_cleaned)
        
        elif strategy == 'fill':
            # Fill missing values based on column type
            fill_info = {}
            
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        if numeric_strategy == 'mean':
                            fill_value = df_cleaned[col].mean()
                        elif numeric_strategy == 'median':
                            fill_value = df_cleaned[col].median()
                        elif numeric_strategy == 'mode':
                            fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 0
                        elif numeric_strategy == 'forward_fill':
                            df_cleaned[col] = df_cleaned[col].ffill().bfill()
                            fill_value = 'forward_fill'
                        else:
                            fill_value = 0
                        
                        if numeric_strategy != 'forward_fill':
                            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                        
                        fill_info[col] = {'strategy': numeric_strategy, 'value': float(fill_value) if isinstance(fill_value, (int, float)) else str(fill_value)}
                    else:
                        # Categorical columns
                        if categorical_strategy == 'mode':
                            fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                        elif categorical_strategy == 'forward_fill':
                            df_cleaned[col] = df_cleaned[col].ffill().bfill()
                            fill_value = 'forward_fill'
                        else:
                            fill_value = 'Unknown'
                        
                        if categorical_strategy != 'forward_fill':
                            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                        
                        fill_info[col] = {'strategy': categorical_strategy, 'value': str(fill_value)}
            
            missing_info['fill_info'] = fill_info
        
        # Final missing values count
        missing_info['after'] = {
            'total_missing': int(df_cleaned.isnull().sum().sum()),
            'columns_with_missing': int((df_cleaned.isnull().sum() > 0).sum())
        }
        
        self.cleaning_report['missing_values'] = missing_info
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None, 
                 keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows."""
        rows_before = len(df)
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        rows_after = len(df_cleaned)
        
        self.cleaning_report['duplicates'] = {
            'removed': rows_before - rows_after,
            'remaining': rows_after
        }
        
        return df_cleaned
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers in numeric columns.
        
        Parameters:
        - method: 'iqr' (Interquartile Range) or 'zscore'
        - threshold: Multiplier for IQR or Z-score threshold
        """
        outliers = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_indices = df[z_scores > threshold].index.tolist()
            else:
                continue
            
            if outlier_indices:
                outliers[col] = outlier_indices
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap', 
                       outlier_method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers.
        
        Parameters:
        - method: 'cap' (cap at bounds), 'remove' (remove rows), or 'transform' (log transform)
        - outlier_method: 'iqr' or 'zscore'
        """
        df_cleaned = df.copy()
        outliers = self.detect_outliers(df_cleaned, method=outlier_method, threshold=threshold)
        
        outlier_info = {}
        
        for col, outlier_indices in outliers.items():
            if method == 'cap':
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_info[col] = {'method': 'cap', 'count': len(outlier_indices), 
                                    'lower_bound': float(lower_bound), 'upper_bound': float(upper_bound)}
            
            elif method == 'remove':
                df_cleaned = df_cleaned.drop(index=outlier_indices)
                outlier_info[col] = {'method': 'remove', 'count': len(outlier_indices)}
            
            elif method == 'transform':
                # Log transform (only for positive values)
                if (df_cleaned[col] > 0).all():
                    df_cleaned[col] = np.log1p(df_cleaned[col])
                    outlier_info[col] = {'method': 'log_transform', 'count': len(outlier_indices)}
        
        self.cleaning_report['outliers'] = outlier_info
        return df_cleaned
    
    def clean(self, df: pd.DataFrame, fix_types: bool = True, 
             handle_missing: bool = True, remove_duplicates: bool = True,
             handle_outliers: bool = True, **kwargs) -> pd.DataFrame:
        """Main cleaning method that applies all cleaning steps."""
        df_cleaned = df.copy()
        
        if fix_types:
            df_cleaned = self.fix_wrong_types(df_cleaned)
        
        if handle_missing:
            df_cleaned = self.handle_missing_values(df_cleaned, **kwargs)
        
        if remove_duplicates:
            df_cleaned = self.remove_duplicates(df_cleaned)
        
        if handle_outliers:
            df_cleaned = self.handle_outliers(df_cleaned, **kwargs)
        
        return df_cleaned
    
    def get_cleaning_report(self) -> Dict:
        """Get detailed cleaning report."""
        return self.cleaning_report

