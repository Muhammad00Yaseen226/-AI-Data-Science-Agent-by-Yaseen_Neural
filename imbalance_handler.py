"""
Imbalance Handler Module
Detects and handles class imbalance using SMOTE and other techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import train_test_split


class ImbalanceHandler:
    """Handles class imbalance in classification datasets."""
    
    def __init__(self):
        self.imbalance_info = {}
        self.resampler = None
        self.resampling_method = None
    
    def detect_imbalance(self, y: pd.Series, threshold: float = 0.4) -> Dict:
        """
        Detect class imbalance.
        
        Parameters:
        - y: Target variable
        - threshold: Imbalance ratio threshold (minority/majority)
        
        Returns:
        - Dictionary with imbalance information
        """
        class_counts = Counter(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        
        if n_classes < 2:
            return {
                'is_imbalanced': False,
                'reason': 'Less than 2 classes',
                'n_classes': n_classes
            }
        
        # Calculate class distribution
        class_distribution = {str(k): int(v) for k, v in class_counts.items()}
        percentages = {str(k): float(v / total_samples * 100) for k, v in class_counts.items()}
        
        # Find majority and minority classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        majority_count = sorted_classes[0][1]
        minority_count = sorted_classes[-1][1]
        imbalance_ratio = minority_count / majority_count if majority_count > 0 else 0
        
        is_imbalanced = imbalance_ratio < threshold
        
        self.imbalance_info = {
            'is_imbalanced': is_imbalanced,
            'n_classes': n_classes,
            'imbalance_ratio': float(imbalance_ratio),
            'threshold': threshold,
            'class_distribution': class_distribution,
            'class_percentages': percentages,
            'majority_class': str(sorted_classes[0][0]),
            'majority_count': int(majority_count),
            'minority_class': str(sorted_classes[-1][0]),
            'minority_count': int(minority_count)
        }
        
        return self.imbalance_info
    
    def apply_resampling(self, X: pd.DataFrame, y: pd.Series, 
                        method: str = 'auto', random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling to balance classes.
        
        Parameters:
        - X: Features
        - y: Target variable
        - method: 'auto', 'smote', 'adasyn', 'random_oversample', 'smote_tomek', 'smote_enn'
        - random_state: Random seed
        
        Returns:
        - Resampled X and y
        """
        imbalance_info = self.detect_imbalance(y)
        
        if not imbalance_info['is_imbalanced']:
            self.resampling_method = 'none'
            return X.copy(), y.copy()
        
        # Auto method selection
        if method == 'auto':
            n_samples = len(X)
            n_features = X.shape[1]
            
            # SMOTE works best with moderate datasets
            if n_samples < 1000:
                method = 'random_oversample'
            elif n_features > 100:
                method = 'random_oversample'  # SMOTE can be slow with many features
            else:
                method = 'smote'
        
        try:
            if method == 'smote':
                resampler = SMOTE(random_state=random_state, n_jobs=-1)
            elif method == 'adasyn':
                resampler = ADASYN(random_state=random_state, n_jobs=-1)
            elif method == 'random_oversample':
                resampler = RandomOverSampler(random_state=random_state)
            elif method == 'smote_tomek':
                resampler = SMOTETomek(random_state=random_state, n_jobs=-1)
            elif method == 'smote_enn':
                resampler = SMOTEENN(random_state=random_state, n_jobs=-1)
            else:
                resampler = SMOTE(random_state=random_state, n_jobs=-1)
            
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns, index=range(len(X_resampled)))
            else:
                X_resampled = pd.DataFrame(X_resampled)
            
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            else:
                y_resampled = pd.Series(y_resampled)
            
            self.resampler = resampler
            self.resampling_method = method
            
            # Update imbalance info after resampling
            new_imbalance_info = self.detect_imbalance(y_resampled)
            self.imbalance_info['after_resampling'] = new_imbalance_info
            
            return X_resampled, y_resampled
            
        except Exception as e:
            # Fallback to random oversampling if SMOTE fails
            if method != 'random_oversample':
                print(f"Warning: {method} failed ({str(e)}). Falling back to random oversampling.")
                resampler = RandomOverSampler(random_state=random_state)
                X_resampled, y_resampled = resampler.fit_resample(X, y)
                
                if isinstance(X, pd.DataFrame):
                    X_resampled = pd.DataFrame(X_resampled, columns=X.columns, index=range(len(X_resampled)))
                else:
                    X_resampled = pd.DataFrame(X_resampled)
                
                if isinstance(y, pd.Series):
                    y_resampled = pd.Series(y_resampled, name=y.name)
                else:
                    y_resampled = pd.Series(y_resampled)
                
                self.resampler = resampler
                self.resampling_method = 'random_oversample'
                
                new_imbalance_info = self.detect_imbalance(y_resampled)
                self.imbalance_info['after_resampling'] = new_imbalance_info
                
                return X_resampled, y_resampled
            else:
                raise
    
    def get_imbalance_info(self) -> Dict:
        """Get imbalance detection and handling information."""
        return self.imbalance_info

