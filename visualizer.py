"""
Visualization Module
Creates histograms, heatmaps, boxplots, pairplots, and other visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Creates visualizations for data analysis."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.created_plots = []
    
    def plot_histograms(self, df: pd.DataFrame, columns: List[str] = None, 
                       bins: int = 30, figsize: tuple = (15, 10)) -> str:
        """Create histograms for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if columns is None:
            columns = numeric_cols.tolist()
        else:
            columns = [col for col in columns if col in numeric_cols]
        
        if not columns:
            return None
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if len(columns) > 1 else [axes]
        
        for idx, col in enumerate(columns):
            ax = axes[idx] if len(columns) > 1 else axes[0]
            df[col].hist(bins=bins, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'Histogram: {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = self.output_dir / "histograms.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_plots.append(str(filepath))
        return str(filepath)
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, figsize: tuple = (12, 10)) -> str:
        """Create correlation heatmap for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filepath = self.output_dir / "correlation_heatmap.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_plots.append(str(filepath))
        return str(filepath)
    
    def plot_boxplots(self, df: pd.DataFrame, columns: List[str] = None,
                     figsize: tuple = (15, 10)) -> str:
        """Create boxplots for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if columns is None:
            columns = numeric_cols.tolist()
        else:
            columns = [col for col in columns if col in numeric_cols]
        
        if not columns:
            return None
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if len(columns) > 1 else [axes]
        
        for idx, col in enumerate(columns):
            ax = axes[idx] if len(columns) > 1 else axes[0]
            df.boxplot(column=col, ax=ax, grid=True)
            ax.set_title(f'Boxplot: {col}', fontsize=12, fontweight='bold')
            ax.set_ylabel(col, fontsize=10)
            plt.setp(ax.get_xticklabels(), visible=False)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = self.output_dir / "boxplots.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_plots.append(str(filepath))
        return str(filepath)
    
    def plot_pairplot(self, df: pd.DataFrame, columns: List[str] = None,
                     sample_size: int = 1000, figsize: tuple = None) -> str:
        """Create pairplot for numeric columns (sampled if too large)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if columns is None:
            columns = numeric_cols.tolist()[:6]  # Limit to 6 columns for readability
        else:
            columns = [col for col in columns if col in numeric_cols][:6]
        
        if len(columns) < 2:
            return None
        
        # Sample data if too large
        df_sample = df[columns].copy()
        if len(df_sample) > sample_size:
            df_sample = df_sample.sample(n=sample_size, random_state=42)
        
        try:
            pairplot = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.6})
            pairplot.fig.suptitle('Pairplot of Numeric Features', y=1.02, fontsize=14, fontweight='bold')
            
            filepath = self.output_dir / "pairplot.png"
            pairplot.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.created_plots.append(str(filepath))
            return str(filepath)
        except Exception as e:
            print(f"Error creating pairplot: {str(e)}")
            return None
    
    def plot_missing_values(self, df: pd.DataFrame, figsize: tuple = (12, 6)) -> str:
        """Visualize missing values."""
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) == 0:
            return None
        
        plt.figure(figsize=figsize)
        missing_counts.plot(kind='barh', color='coral', edgecolor='black')
        plt.title('Missing Values by Column', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Count of Missing Values', fontsize=11)
        plt.ylabel('Columns', fontsize=11)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        filepath = self.output_dir / "missing_values.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_plots.append(str(filepath))
        return str(filepath)
    
    def plot_categorical_counts(self, df: pd.DataFrame, columns: List[str] = None,
                               top_n: int = 10, figsize: tuple = (15, 10)) -> str:
        """Plot value counts for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if columns is None:
            columns = categorical_cols.tolist()[:6]  # Limit to 6 columns
        else:
            columns = [col for col in columns if col in categorical_cols][:6]
        
        if not columns:
            return None
        
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if len(columns) > 1 else [axes]
        
        for idx, col in enumerate(columns):
            ax = axes[idx] if len(columns) > 1 else axes[0]
            value_counts = df[col].value_counts().head(top_n)
            value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Top {top_n} Values: {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = self.output_dir / "categorical_counts.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_plots.append(str(filepath))
        return str(filepath)
    
    def plot_distribution_comparison(self, df: pd.DataFrame, columns: List[str] = None,
                                    figsize: tuple = (15, 10)) -> str:
        """Compare distributions using KDE plots."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if columns is None:
            columns = numeric_cols.tolist()[:6]
        else:
            columns = [col for col in columns if col in numeric_cols][:6]
        
        if not columns:
            return None
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if len(columns) > 1 else [axes]
        
        for idx, col in enumerate(columns):
            ax = axes[idx] if len(columns) > 1 else axes[0]
            df[col].plot(kind='kde', ax=ax, color='darkblue', linewidth=2)
            ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label='Median')
            ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = self.output_dir / "distributions.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_plots.append(str(filepath))
        return str(filepath)
    
    def create_all_visualizations(self, df: pd.DataFrame) -> List[str]:
        """Create all available visualizations."""
        print("Creating visualizations...")
        
        plots_created = []
        
        # Histograms
        try:
            plot_path = self.plot_histograms(df)
            if plot_path:
                plots_created.append(plot_path)
                print("  ✓ Histograms created")
        except Exception as e:
            print(f"  ✗ Error creating histograms: {str(e)}")
        
        # Correlation heatmap
        try:
            plot_path = self.plot_correlation_heatmap(df)
            if plot_path:
                plots_created.append(plot_path)
                print("  ✓ Correlation heatmap created")
        except Exception as e:
            print(f"  ✗ Error creating correlation heatmap: {str(e)}")
        
        # Boxplots
        try:
            plot_path = self.plot_boxplots(df)
            if plot_path:
                plots_created.append(plot_path)
                print("  ✓ Boxplots created")
        except Exception as e:
            print(f"  ✗ Error creating boxplots: {str(e)}")
        
        # Pairplot
        try:
            plot_path = self.plot_pairplot(df)
            if plot_path:
                plots_created.append(plot_path)
                print("  ✓ Pairplot created")
        except Exception as e:
            print(f"  ✗ Error creating pairplot: {str(e)}")
        
        # Missing values
        try:
            plot_path = self.plot_missing_values(df)
            if plot_path:
                plots_created.append(plot_path)
                print("  ✓ Missing values plot created")
        except Exception as e:
            print(f"  ✗ Error creating missing values plot: {str(e)}")
        
        # Categorical counts
        try:
            plot_path = self.plot_categorical_counts(df)
            if plot_path:
                plots_created.append(plot_path)
                print("  ✓ Categorical counts plot created")
        except Exception as e:
            print(f"  ✗ Error creating categorical counts plot: {str(e)}")
        
        # Distribution comparison
        try:
            plot_path = self.plot_distribution_comparison(df)
            if plot_path:
                plots_created.append(plot_path)
                print("  ✓ Distribution plots created")
        except Exception as e:
            print(f"  ✗ Error creating distribution plots: {str(e)}")
        
        return plots_created
    
    def get_created_plots(self) -> List[str]:
        """Get list of created plot file paths."""
        return self.created_plots

