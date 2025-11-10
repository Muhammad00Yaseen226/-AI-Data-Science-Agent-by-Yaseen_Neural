"""
Main AI Data Cleaning & EDA Agent
Orchestrates all modules to provide end-to-end data cleaning and analysis
"""

import os
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from file_handler import FileHandler
from data_cleaner import DataCleaner
from preprocessor import Preprocessor
from eda_analyzer import EDAAnalyzer
from visualizer import Visualizer
from automl import run_automl_pipeline


class DataCleaningAgent:
    """Main agent that orchestrates data cleaning, preprocessing, EDA, and visualization."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_handler = FileHandler()
        self.data_cleaner = DataCleaner()
        self.preprocessor = Preprocessor()
        self.eda_analyzer = EDAAnalyzer()
        self.visualizer = Visualizer(output_dir=str(self.output_dir / "plots"))
        
        self.raw_data = None
        self.cleaned_data = None
        self.processed_data = None
        self.eda_report = None
    
    def load_data(self, input_path: str, **kwargs) -> pd.DataFrame:
        print(f"\n{'='*80}")
        print(f"📂 Loading data from: {input_path}")
        print(f"{'='*80}")
        
        try:
            self.raw_data = self.file_handler.load_data(input_path, **kwargs)
            print(f"✓ Successfully loaded data")
            print(f"  Shape: {self.raw_data.shape[0]} rows × {self.raw_data.shape[1]} columns")
            print(f"  Columns: {', '.join(self.raw_data.columns.tolist()[:5])}{'...' if len(self.raw_data.columns) > 5 else ''}")
            return self.raw_data
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def clean_data(self, **kwargs) -> pd.DataFrame:
        if self.raw_data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        print(f"\n{'='*80}\n🧹 Cleaning data...\n{'='*80}")
        self.cleaned_data = self.data_cleaner.clean(self.raw_data, **kwargs)
        cleaning_report = self.data_cleaner.get_cleaning_report()
        
        print("✓ Data cleaning completed")
        if 'missing_values' in cleaning_report:
            mv = cleaning_report['missing_values']
            print(f"  Missing values handled: {mv.get('before', {}).get('total_missing', 0)} → {mv.get('after', {}).get('total_missing', 0)}")
        if 'duplicates' in cleaning_report:
            dup = cleaning_report['duplicates']
            print(f"  Duplicates removed: {dup.get('removed', 0)}")
        if 'outliers' in cleaning_report:
            outliers = cleaning_report['outliers']
            print(f"  Outliers handled in {len(outliers)} columns")
        
        return self.cleaned_data
    
    def preprocess_data(self, **kwargs) -> pd.DataFrame:
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        
        print(f"\n{'='*80}\n⚙️  Preprocessing data...\n{'='*80}")
        self.processed_data = self.preprocessor.preprocess(self.cleaned_data, **kwargs)
        preprocessing_info = self.preprocessor.get_preprocessing_info()
        
        print("✓ Data preprocessing completed")
        if 'encoding' in preprocessing_info:
            encoding = preprocessing_info['encoding']
            print(f"  Categorical columns encoded: {len(encoding)}")
        if 'scaling' in preprocessing_info:
            scaling = preprocessing_info['scaling']
            print(f"  Numeric columns scaled: {len(scaling)}")
        
        return self.processed_data
    
    def perform_eda(self) -> Dict:
        data_for_eda = self.processed_data if self.processed_data is not None else self.cleaned_data
        if data_for_eda is None:
            raise ValueError("No data available for EDA. Please load and clean data first.")
        
        print(f"\n{'='*80}\n📊 Performing Exploratory Data Analysis...\n{'='*80}")
        self.eda_report = self.eda_analyzer.generate_report(data_for_eda)
        print("✓ EDA completed")
        print(f"  Generated statistics for {len(data_for_eda.columns)} columns")
        return self.eda_report
    
    def create_visualizations(self) -> list:
        data_for_viz = self.processed_data if self.processed_data is not None else self.cleaned_data
        if data_for_viz is None:
            raise ValueError("No data available for visualization. Please load and clean data first.")
        
        print(f"\n{'='*80}\n📈 Creating visualizations...\n{'='*80}")
        plots = self.visualizer.create_all_visualizations(data_for_viz)
        print(f"✓ Created {len(plots)} visualization files")
        return plots
    
    def export_results(self, filename_prefix: str = "cleaned_data") -> Dict[str, str]:
        print(f"\n{'='*80}\n💾 Exporting results...\n{'='*80}")
        exported_files = {}
        
        data_to_export = self.processed_data if self.processed_data is not None else self.cleaned_data
        if data_to_export is not None:
            csv_path = self.output_dir / f"{filename_prefix}.csv"
            data_to_export.to_csv(csv_path, index=False)
            exported_files['cleaned_data'] = str(csv_path)
            print(f"✓ Exported cleaned data: {csv_path}")
        
        if self.eda_report is not None:
            report_text = self.eda_analyzer.format_report_text(self.eda_report)
            report_path = self.output_dir / "eda_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            exported_files['eda_report'] = str(report_path)
            print(f"✓ Exported EDA report: {report_path}")
        
        plots = self.visualizer.get_created_plots()
        if plots:
            exported_files['visualizations'] = plots
            print(f"✓ Visualizations saved in: {self.output_dir / 'plots'}")
        
        print(f"\n{'='*80}\n✅ All results exported successfully!\n{'='*80}")
        return exported_files
    
    def run_full_pipeline(self, input_path: str, clean: bool = True, 
                         preprocess: bool = True, eda: bool = True,
                         visualize: bool = True, export: bool = True,
                         run_automl: bool = False,
                         automl_target: str = None,
                         **kwargs) -> Dict:
        print("\n" + "="*80)
        print("🚀 AI DATA CLEANING & EDA AGENT - FULL PIPELINE")
        print("="*80)
        
        results = {}
        try:
            self.load_data(input_path, **kwargs)
            results['raw_shape'] = self.raw_data.shape
            
            if clean:
                self.clean_data(**kwargs)
                results['cleaned_shape'] = self.cleaned_data.shape
            if preprocess:
                self.preprocess_data(**kwargs)
                results['processed_shape'] = self.processed_data.shape
            if eda:
                self.perform_eda()
                results['eda_report'] = self.eda_report
            if visualize:
                plots = self.create_visualizations()
                results['visualizations'] = plots
            
            # --- AutoML section ---
            if run_automl:
                print(f"\n{'='*80}\n🤖 Running AutoML pipeline...\n{'='*80}")
                data_for_model = self.processed_data if self.processed_data is not None else self.cleaned_data
                if data_for_model is None:
                    raise ValueError("No data available for AutoML. Please run clean/preprocess first.")
                
                automl_results = run_automl_pipeline(
                    data_for_model,
                    output_dir=str(self.output_dir),
                    target_col=automl_target
                )
                results['automl'] = automl_results
                print(f"✓ AutoML completed. Best model: {automl_results.get('best_model_name')}")
            
            if export:
                exported = self.export_results()
                results['exported_files'] = exported
            
            print("\n" + "="*80)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            return results
        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {str(e)}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Data Cleaning & EDA Agent')
    parser.add_argument('input_path', type=str, help='Path to input file or folder')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--no-clean', action='store_true', help='Skip data cleaning')
    parser.add_argument('--no-preprocess', action='store_true', help='Skip preprocessing')
    parser.add_argument('--no-eda', action='store_true', help='Skip EDA')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualizations')
    parser.add_argument('--no-export', action='store_true', help='Skip export')

    args = parser.parse_args()
    
    # Hardcoded AutoML
    AUTOML_TARGET = "PurchaseAmount"
    
    agent = DataCleaningAgent(output_dir=args.output_dir)
    agent.run_full_pipeline(
        input_path=args.input_path,
        clean=not args.no_clean,
        preprocess=not args.no_preprocess,
        eda=not args.no_eda,
        visualize=not args.no_visualize,
        export=not args.no_export,
        run_automl=True,         # always run AutoML
        automl_target=AUTOML_TARGET
    )


if __name__ == "__main__":
    main()
