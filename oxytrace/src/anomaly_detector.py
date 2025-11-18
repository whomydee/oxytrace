"""
Anomaly Detection Module for Oxygen Sensor Data
"""

from pathlib import Path
import pickle

import pandas as pd

from oxytrace.src.models.statstical_detector import StatisticalDetector
from oxytrace.src.utils.custom_logger import LOGGER



class AnomalyDetector:
    """
    Ensemble anomaly detector combining multiple detection strategies.
    
    Detects:
    - Point anomalies (statistical outliers)
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
    ):
        """
        Initialize ensemble detector.
        
        Args:
            z_threshold: Z-score threshold for statistical detection
        """
        self.statistical_detector = StatisticalDetector(z_threshold=z_threshold)
        
        self.weights = {
            'statistical': 0.25,
            # I will add more weights here as I have new models
        }
    
    def fit(
        self,
        df: pd.DataFrame,
        value_col: str = 'Oxygen[%sat]',
    ):
        """
        Train detector on normal data.
        
        Args:
            df: Training dataframe with preprocessed features
            value_col: Name of the oxygen value column
        """
        LOGGER.info("Training anomaly detector", rows=len(df))
        
        values = df[value_col].values
        
        # Train statistica detector
        self.statistical_detector.fit(values)
        
        LOGGER.info("Anomaly detector trained successfully")
    
    def detect(
        self,
        df: pd.DataFrame,
        value_col: str = 'Oxygen[%sat]'
    ) -> pd.DataFrame:
        """
        Detect anomalies using statistical method.
        
        Args:
            df: Dataframe to analyze
            value_col: Name of the oxygen value column
            
        Returns:
            DataFrame with anomaly detection results
        """
        values = df[value_col].values
        results_df = df.copy()
        
        # Get detections from statistical detector
        stat_anom, stat_scores = self.statistical_detector.detect(values)
        
        # Add detection results to dataframe
        results_df['is_anomaly'] = stat_anom
        results_df['anomaly_score'] = stat_scores
        
        # Categorize anomaly type
        results_df['anomaly_type'] = results_df['is_anomaly'].apply(
            lambda x: 'point' if x else 'normal'
        )
        
        LOGGER.info(
            "Anomaly detection complete",
            total_records=len(results_df),
            anomalies_detected=results_df['is_anomaly'].sum(),
            anomaly_rate=f"{results_df['is_anomaly'].mean() * 100:.2f}%"
        )
        
        return results_df
    
    def save(self, filepath: str):
        """Save detector models to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'statistical_detector': self.statistical_detector,
            'weights': self.weights
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        LOGGER.info("Anomaly detector saved", filepath=str(filepath))
    
    def load(self, filepath: str):
        """Load detector models from disk."""
        filepath = Path(filepath)
        
        # Load other components
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.statistical_detector = save_dict['statistical_detector']
        self.weights = save_dict['weights']
        
        LOGGER.info("Anomaly detector loaded", filepath=str(filepath))
