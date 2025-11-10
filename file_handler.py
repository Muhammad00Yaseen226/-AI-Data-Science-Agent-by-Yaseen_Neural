"""
File Handler Module
Handles reading different file types: CSV, Excel, JSON, TXT, ZIP
"""

import os
import pandas as pd
import zipfile
import json
from pathlib import Path
from typing import Union, List, Optional


class FileHandler:
    """Handles reading and merging data from various file formats."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.txt', '.zip']
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        ext = Path(file_path).suffix.lower()
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {self.supported_formats}")
        return ext
    
    def read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file."""
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {str(e)}")
    
    def read_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Excel file."""
        try:
            return pd.read_excel(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading Excel file {file_path}: {str(e)}")
    
    def read_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find the main data key
                if 'data' in data:
                    return pd.DataFrame(data['data'])
                else:
                    return pd.DataFrame([data])
            else:
                return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Error reading JSON file {file_path}: {str(e)}")
    
    def read_txt(self, file_path: str, delimiter: str = None, **kwargs) -> pd.DataFrame:
        """Read text file (assumes tab or comma delimited)."""
        try:
            if delimiter is None:
                # Try common delimiters
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        delimiter = '\t'
                    elif ',' in first_line:
                        delimiter = ','
                    else:
                        delimiter = ' '
            
            return pd.read_csv(file_path, delimiter=delimiter, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading TXT file {file_path}: {str(e)}")
    
    def extract_zip(self, zip_path: str, extract_to: str = None) -> List[str]:
        """Extract ZIP file and return list of extracted file paths."""
        if extract_to is None:
            extract_to = os.path.join(os.path.dirname(zip_path), 'extracted')
        
        os.makedirs(extract_to, exist_ok=True)
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                extracted_files = [os.path.join(extract_to, f) for f in zip_ref.namelist()]
        except Exception as e:
            raise ValueError(f"Error extracting ZIP file {zip_path}: {str(e)}")
        
        return extracted_files
    
    def read_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read file based on detected type."""
        file_type = self.detect_file_type(file_path)
        
        if file_type == '.csv':
            return self.read_csv(file_path, **kwargs)
        elif file_type in ['.xlsx', '.xls']:
            return self.read_excel(file_path, **kwargs)
        elif file_type == '.json':
            return self.read_json(file_path, **kwargs)
        elif file_type == '.txt':
            return self.read_txt(file_path, **kwargs)
        elif file_type == '.zip':
            # Extract and read all CSV files from ZIP
            extracted_files = self.extract_zip(file_path)
            dataframes = []
            for extracted_file in extracted_files:
                if extracted_file.endswith('.csv'):
                    try:
                        df = self.read_csv(extracted_file)
                        dataframes.append(df)
                    except:
                        continue
            if not dataframes:
                raise ValueError("No CSV files found in ZIP archive")
            return self.merge_dataframes(dataframes)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple dataframes with consistent schema."""
        if not dataframes:
            raise ValueError("No dataframes to merge")
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Find common columns
        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        if not common_cols:
            raise ValueError("No common columns found across dataframes")
        
        # Align columns and merge
        aligned_dfs = []
        for df in dataframes:
            # Keep only common columns
            df_aligned = df[[col for col in df.columns if col in common_cols]].copy()
            aligned_dfs.append(df_aligned)
        
        merged_df = pd.concat(aligned_dfs, ignore_index=True)
        return merged_df
    
    def read_folder(self, folder_path: str) -> pd.DataFrame:
        """Read all supported files from a folder and merge them."""
        dataframes = []
        folder_path = Path(folder_path)
        
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                try:
                    file_type = self.detect_file_type(str(file_path))
                    df = self.read_file(str(file_path))
                    dataframes.append(df)
                except ValueError:
                    # Skip unsupported files
                    continue
        
        if not dataframes:
            raise ValueError(f"No supported files found in folder: {folder_path}")
        
        return self.merge_dataframes(dataframes)
    
    def load_data(self, input_path: str, **kwargs) -> pd.DataFrame:
        """Main method to load data from file or folder."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")
        
        if input_path.is_file():
            return self.read_file(str(input_path), **kwargs)
        elif input_path.is_dir():
            return self.read_folder(str(input_path))
        else:
            raise ValueError(f"Invalid path: {input_path}")

