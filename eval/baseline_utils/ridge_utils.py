from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.linear_model import RidgeClassifier
import numpy as np

class AdaptiveCalibratedRidgeClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for RidgeClassifier with adaptive cross-validation for calibration.
    
    Automatically adjusts cv parameter based on data size to handle small datasets
    in label scarcity scenarios.
    """
    def __init__(self, alpha=1.0, random_state=None, class_weight='balanced'):
        self.alpha = alpha
        self.random_state = random_state
        self.class_weight = class_weight
        self.base_estimator_ = None
        self.calibrated_estimator_ = None
    
    def fit(self, X, y, sample_weight=None):
        # Determine appropriate cv based on minimum class count
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min() if len(counts) > 0 else 0
        
        # For low training portion evaluation
        if min_class_count >= 2:
            cv = 2
        else:
            cv = 1
        
        self.base_estimator_ = RidgeClassifier(
            alpha=self.alpha,
            random_state=self.random_state,
            class_weight=self.class_weight
        )
        
        # Fit base estimator
        self.base_estimator_.fit(X, y, sample_weight=sample_weight)
        
        # Calibrate if we have enough samples
        if len(y) >= 2:
            try:
                self.calibrated_estimator_ = CalibratedClassifierCV(
                    self.base_estimator_,
                    method='sigmoid',
                    cv=cv
                )
                self.calibrated_estimator_.fit(X, y, sample_weight=sample_weight)
            except ValueError:
                self.calibrated_estimator_ = None
        else:
            self.calibrated_estimator_ = None
        
        return self
    
    def predict(self, X):
        return self.base_estimator_.predict(X)
    
    def predict_proba(self, X):
        if self.calibrated_estimator_ is not None:
            return self.calibrated_estimator_.predict_proba(X)
        else:
            decision = self.base_estimator_.decision_function(X)
            proba_positive = 1 / (1 + np.exp(-decision))
            proba_negative = 1 - proba_positive
            return np.column_stack([proba_negative, proba_positive])