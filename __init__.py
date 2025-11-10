"""
AI Data Cleaning & EDA Agent
A comprehensive tool for automatic data cleaning, preprocessing, EDA, and visualization.
"""

from .main_agent import DataCleaningAgent
from .file_handler import FileHandler
from .data_cleaner import DataCleaner
from .preprocessor import Preprocessor
from .eda_analyzer import EDAAnalyzer
from .visualizer import Visualizer

__version__ = "1.0.0"
__all__ = [
    'DataCleaningAgent',
    'FileHandler',
    'DataCleaner',
    'Preprocessor',
    'EDAAnalyzer',
    'Visualizer'
]

