from typing import Tuple
import numpy as np

from oxytrace.src.utils.custom_logger import LOGGER


class StatisticalDetector:
    """Detect point anomalies using statistical methods."""
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize statistical detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
            iqr_multiplier: IQR multiplier for outlier detection
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.mean = None
        self.std = None
        self.q1 = None
        self.q3 = None
        self.iqr = None
    
    def fit(self, values: np.ndarray):
        """Learn normal distribution parameters."""
        self.mean = np.mean(values)
        self.std = np.std(values)
        self.q1 = np.percentile(values, 25)
        self.q3 = np.percentile(values, 75)
        self.iqr = self.q3 - self.q1
        
        LOGGER.info(
            "Statistical detector fitted",
            mean=float(self.mean),
            std=float(self.std),
            q1=float(self.q1),
            q3=float(self.q3)
        )
    
    def detect(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score and IQR methods.
        
        Returns:
            Tuple of (is_anomaly, anomaly_scores)
        """
        # Z-score based detection
        z_scores = np.abs((values - self.mean) / (self.std + 1e-8))
        z_anomalies = z_scores > self.z_threshold
        
        # IQR based detection
        lower_bound = self.q1 - self.iqr_multiplier * self.iqr
        upper_bound = self.q3 + self.iqr_multiplier * self.iqr
        iqr_anomalies = (values < lower_bound) | (values > upper_bound)
        
        # Combine both methods
        is_anomaly = z_anomalies | iqr_anomalies
        
        # Score is normalized z-score (0-100 scale)
        anomaly_scores = np.clip((z_scores / self.z_threshold) * 100, 0, 100)
        
        return is_anomaly, anomaly_scores
