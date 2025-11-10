"""
EDA (Exploratory Data Analysis) Module
Generates statistical summaries and data profiling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class EDAAnalyzer:
    """Performs exploratory data analysis on datasets."""
    
    def __init__(self):
        self.eda_report = {}
    
    def basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistical summary."""
        stats = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage': {
                'total_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
                'per_column_mb': (df.memory_usage(deep=True) / 1024**2).to_dict()
            }
        }
        
        return stats
    
    def numeric_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        summary = df[numeric_cols].describe()
        
        # Add additional statistics
        additional_stats = pd.DataFrame({
            'skewness': df[numeric_cols].skew(),
            'kurtosis': df[numeric_cols].kurtosis(),
            'variance': df[numeric_cols].var(),
            'range': df[numeric_cols].max() - df[numeric_cols].min()
        })
        
        summary = pd.concat([summary, additional_stats])
        return summary
    
    def categorical_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {}
        
        summary = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_5_values': value_counts.head(5).to_dict()
            }
        
        return summary
    
    def missing_values_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': int((missing_counts > 0).sum()),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
    
    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        
        return df[numeric_cols].corr()
    
    def detect_high_correlations(self, df: pd.DataFrame, threshold: float = 0.8) -> List[tuple]:
        """Detect highly correlated feature pairs."""
        corr_matrix = self.correlation_analysis(df)
        
        if corr_matrix.empty:
            return []
        
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_value)
                    ))
        
        return high_corr_pairs
    
    def distribution_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze distributions of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        distribution_info = {}
        
        for col in numeric_cols:
            distribution_info[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'mode': float(df[col].mode()[0]) if not df[col].mode().empty else None,
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'iqr': float(df[col].quantile(0.75) - df[col].quantile(0.25)),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        return distribution_info
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive EDA report."""
        self.eda_report = {
            'basic_statistics': self.basic_statistics(df),
            'numeric_summary': self.numeric_summary(df).to_dict() if not self.numeric_summary(df).empty else {},
            'categorical_summary': self.categorical_summary(df),
            'missing_values': self.missing_values_analysis(df),
            'correlation_matrix': self.correlation_analysis(df).to_dict() if not self.correlation_analysis(df).empty else {},
            'high_correlations': self.detect_high_correlations(df),
            'distribution_analysis': self.distribution_analysis(df)
        }
        
        return self.eda_report
    
    def format_report_text(self, report: Dict = None) -> str:
        """Format EDA report as readable text."""
        if report is None:
            report = self.eda_report
        
        if not report:
            return "No EDA report available."
        
        lines = []
        lines.append("=" * 80)
        lines.append("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Basic Statistics
        if 'basic_statistics' in report:
            bs = report['basic_statistics']
            lines.append("\n📊 BASIC STATISTICS")
            lines.append("-" * 80)
            lines.append(f"Dataset Shape: {bs.get('shape', {}).get('rows', 'N/A')} rows × {bs.get('shape', {}).get('columns', 'N/A')} columns")
            lines.append(f"Total Memory Usage: {bs.get('memory_usage', {}).get('total_mb', 0):.2f} MB")
            lines.append("")
            lines.append("Data Types:")
            for col, dtype in bs.get('data_types', {}).items():
                lines.append(f"  - {col}: {dtype}")
        
        # Missing Values
        if 'missing_values' in report:
            mv = report['missing_values']
            lines.append("\n\n🔍 MISSING VALUES ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Total Missing Values: {mv.get('total_missing', 0)}")
            lines.append(f"Columns with Missing Values: {mv.get('columns_with_missing', 0)}")
            if mv.get('missing_by_column'):
                lines.append("\nMissing Values by Column:")
                for col, count in mv['missing_by_column'].items():
                    pct = mv.get('missing_percentages', {}).get(col, 0)
                    lines.append(f"  - {col}: {count} ({pct:.2f}%)")
        
        # Numeric Summary
        if 'numeric_summary' in report and report['numeric_summary']:
            lines.append("\n\n📈 NUMERIC COLUMNS SUMMARY")
            lines.append("-" * 80)
            # Format numeric summary as table would be complex, so we'll use distribution analysis instead
        
        # Distribution Analysis
        if 'distribution_analysis' in report and report['distribution_analysis']:
            lines.append("\n\n📊 DISTRIBUTION ANALYSIS")
            lines.append("-" * 80)
            for col, stats in report['distribution_analysis'].items():
                lines.append(f"\n{col}:")
                lines.append(f"  Mean: {stats.get('mean', 'N/A'):.4f}")
                lines.append(f"  Median: {stats.get('median', 'N/A'):.4f}")
                lines.append(f"  Std Dev: {stats.get('std', 'N/A'):.4f}")
                lines.append(f"  Range: [{stats.get('min', 'N/A'):.4f}, {stats.get('max', 'N/A'):.4f}]")
                lines.append(f"  IQR: {stats.get('iqr', 'N/A'):.4f}")
                lines.append(f"  Skewness: {stats.get('skewness', 'N/A'):.4f}")
                lines.append(f"  Kurtosis: {stats.get('kurtosis', 'N/A'):.4f}")
        
        # Categorical Summary
        if 'categorical_summary' in report and report['categorical_summary']:
            lines.append("\n\n📋 CATEGORICAL COLUMNS SUMMARY")
            lines.append("-" * 80)
            for col, stats in report['categorical_summary'].items():
                lines.append(f"\n{col}:")
                lines.append(f"  Unique Values: {stats.get('unique_count', 'N/A')}")
                lines.append(f"  Most Frequent: {stats.get('most_frequent', 'N/A')} (count: {stats.get('most_frequent_count', 'N/A')})")
                if stats.get('top_5_values'):
                    lines.append("  Top 5 Values:")
                    for val, count in list(stats['top_5_values'].items())[:5]:
                        lines.append(f"    - {val}: {count}")
        
        # High Correlations
        if 'high_correlations' in report and report['high_correlations']:
            lines.append("\n\n🔗 HIGH CORRELATIONS (>0.8)")
            lines.append("-" * 80)
            for col1, col2, corr in report['high_correlations']:
                lines.append(f"  {col1} ↔ {col2}: {corr:.4f}")
        
        lines.append("\n" + "=" * 80)
        lines.append("End of EDA Report")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_report(self) -> Dict:
        """Get the EDA report."""
        return self.eda_report

