"""
Preprocessing Module
Handles categorical encoding, scaling, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from typing import List, Optional, Dict


class Preprocessor:
    """Preprocesses data for machine learning models."""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.preprocessing_info = {}
    
    def identify_categorical(self, df: pd.DataFrame, max_unique: int = 20) -> List[str]:
        """Identify categorical columns."""
        categorical_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'int32'] and df[col].nunique() <= max_unique:
                # Integer columns with few unique values might be categorical
                categorical_cols.append(col)
        
        return categorical_cols
    
    def identify_numeric(self, df: pd.DataFrame) -> List[str]:
        """Identify numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    def encode_categorical(self, df: pd.DataFrame, method: str = 'auto',
                          columns: List[str] = None, max_categories: int = 10) -> pd.DataFrame:
        """
        Encode categorical columns.
        
        Parameters:
        - method: 'auto', 'onehot', 'label', or 'ordinal'
        - columns: Specific columns to encode (None = auto-detect)
        - max_categories: Max categories for one-hot encoding
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = self.identify_categorical(df_processed, max_unique=max_categories)
        
        encoding_info = {}
        
        for col in columns:
            if col not in df_processed.columns:
                continue
            
            n_unique = df_processed[col].nunique()
            
            # Auto method: one-hot for few categories, label for many
            if method == 'auto':
                if n_unique <= max_categories:
                    encode_method = 'onehot'
                else:
                    encode_method = 'label'
            else:
                encode_method = method
            
            try:
                if encode_method == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                    df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
                    encoding_info[col] = {'method': 'onehot', 'new_columns': dummies.columns.tolist()}
                
                elif encode_method == 'label':
                    # Label encoding
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.encoders[col] = le
                    encoding_info[col] = {'method': 'label', 'n_categories': n_unique}
                
                elif encode_method == 'ordinal':
                    # Ordinal encoding
                    oe = OrdinalEncoder()
                    df_processed[[col]] = oe.fit_transform(df_processed[[col]])
                    self.encoders[col] = oe
                    encoding_info[col] = {'method': 'ordinal', 'n_categories': n_unique}
            
            except Exception as e:
                encoding_info[col] = {'method': 'failed', 'error': str(e)}
        
        self.preprocessing_info['encoding'] = encoding_info
        return df_processed
    
    def scale_numeric(self, df: pd.DataFrame, method: str = 'standard',
                     columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numeric columns.
        
        Parameters:
        - method: 'standard', 'minmax', 'robust', or None
        - columns: Specific columns to scale (None = all numeric)
        """
        if method is None:
            return df
        
        df_processed = df.copy()
        
        if columns is None:
            columns = self.identify_numeric(df_processed)
        
        if not columns:
            return df_processed
        
        scaling_info = {}
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return df_processed
        
        # Scale columns
        for col in columns:
            if col in df_processed.columns:
                try:
                    col_scaler = type(scaler)(copy=True)
                    df_processed[[col]] = col_scaler.fit_transform(df_processed[[col]])
                    self.scalers[col] = col_scaler
                    scaling_info[col] = {'method': method, 'mean': float(df_processed[col].mean()), 
                                        'std': float(df_processed[col].std())}
                except Exception as e:
                    scaling_info[col] = {'method': 'failed', 'error': str(e)}
        
        self.preprocessing_info['scaling'] = scaling_info
        return df_processed
    
    def normalize(self, df: pd.DataFrame, method: str = 'l2',
                 columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize data (L1 or L2 normalization).
        
        Parameters:
        - method: 'l1' or 'l2'
        - columns: Specific columns to normalize (None = all numeric)
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = self.identify_numeric(df_processed)
        
        if not columns:
            return df_processed
        
        from sklearn.preprocessing import normalize
        
        for col in columns:
            if col in df_processed.columns:
                try:
                    values = df_processed[[col]].values
                    normalized = normalize(values, norm=method, axis=0)
                    df_processed[col] = normalized.flatten()
                except Exception as e:
                    continue
        
        return df_processed
    
    def preprocess(self, df: pd.DataFrame, encode_categorical: bool = True,
                  scale_numeric: bool = True, normalize_data: bool = False,
                  encoding_method: str = 'auto', scaling_method: str = 'standard',
                  **kwargs) -> pd.DataFrame:
        """Main preprocessing method."""
        df_processed = df.copy()
        
        if encode_categorical:
            df_processed = self.encode_categorical(df_processed, method=encoding_method, **kwargs)
        
        if scale_numeric:
            df_processed = self.scale_numeric(df_processed, method=scaling_method, **kwargs)
        
        if normalize_data:
            df_processed = self.normalize(df_processed, **kwargs)
        
        return df_processed
    
    def get_preprocessing_info(self) -> Dict:
        """Get preprocessing information."""
        return self.preprocessing_info

