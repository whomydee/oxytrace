"""
Feature Engineering for Oxygen Sensor Data

Transforms raw oxygen readings into features that help detect different
types of anomalies without needing multiple specialized models.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class OxygenFeatureEngineer:
    """Engineer features from raw oxygen sensor data."""
    
    def __init__(self, short_window=60, long_window=360, fault_window=30):
        self.short_window = short_window
        self.long_window = long_window
        self.fault_window = fault_window
        self.global_mean_ = None
        self.global_std_ = None
    
    def fit(self, df, oxygen_col='Oxygen[%sat]'):
        """Learn global statistics from the data."""
        oxygen = df[oxygen_col].dropna()
        self.global_mean_ = oxygen.mean()
        self.global_std_ = oxygen.std()
        return self
    
    def transform(self, df, oxygen_col='Oxygen[%sat]'):
        """Transform oxygen data into features."""
        if self.global_mean_ is None:
            raise RuntimeError("Must call fit() before transform()")
        
        oxygen = df[oxygen_col]
        features = pd.DataFrame(index=df.index)
        
        # Raw value
        features['oxygen_value'] = oxygen
        
        # Short-term rolling stats (1 hour window)
        features['rolling_mean_60'] = oxygen.rolling(self.short_window, min_periods=1).mean()
        features['rolling_std_60'] = oxygen.rolling(self.short_window, min_periods=1).std()
        features['rolling_median_60'] = oxygen.rolling(self.short_window, min_periods=1).median()
        features['rolling_min_60'] = oxygen.rolling(self.short_window, min_periods=1).min()
        
        # Long-term rolling stats (6 hour window)
        features['rolling_mean_360'] = oxygen.rolling(self.long_window, min_periods=1).mean()
        features['rolling_std_360'] = oxygen.rolling(self.long_window, min_periods=1).std()
        
        # Rate of change at different scales
        features['rate_change_1min'] = oxygen.diff(1).abs()
        features['rate_change_5min'] = oxygen.diff(5).abs()
        features['rate_change_60min'] = oxygen.diff(60).abs()
        
        # Statistical deviations
        features['z_score_global'] = (oxygen - self.global_mean_) / self.global_std_
        features['z_score_local'] = (oxygen - features['rolling_mean_60']) / (features['rolling_std_60'] + 1e-6)
        features['dist_from_median'] = (oxygen - features['rolling_median_60']).abs()
        
        # Sensor fault indicators
        features['variance_30min'] = oxygen.rolling(self.fault_window, min_periods=1).var()
        features['consecutive_same'] = self._count_consecutive_same(oxygen)
        features['range_30min'] = (oxygen.rolling(self.fault_window, min_periods=1).max() - 
                                   oxygen.rolling(self.fault_window, min_periods=1).min())
        features['coeff_variation'] = features['rolling_std_60'] / (features['rolling_mean_60'] + 1e-6)
        
        # Enhanced stuck sensor detection
        features['max_consecutive_same_60'] = self._max_consecutive_same_window(oxygen, self.short_window)
        features['value_change_rate'] = oxygen.diff().abs().rolling(self.fault_window, min_periods=1).mean()
        
        # Noisy sensor detection
        features['point_to_point_variance'] = oxygen.diff().rolling(self.fault_window, min_periods=1).var()
        features['high_freq_energy'] = self._compute_high_freq_energy(oxygen, self.fault_window)
        features['outlier_density_30min'] = self._compute_outlier_density(oxygen, features['rolling_mean_60'], 
                                                                          features['rolling_std_60'], self.fault_window)
        
        # Collective anomaly detection
        features['autocorr_lag5'] = self._compute_autocorr(oxygen, lag=5, window=self.short_window)
        features['trend_strength_60min'] = self._compute_trend_strength(oxygen, self.short_window)
        features['pattern_deviation'] = self._compute_pattern_deviation(oxygen, features['rolling_mean_360'], 
                                                                        features['rolling_std_360'])
        
        # Fill NaNs from rolling operations
        features = features.ffill().bfill()
        
        return features
    
    def fit_transform(self, df, oxygen_col='Oxygen[%sat]'):
        return self.fit(df, oxygen_col).transform(df, oxygen_col)
    
    def _count_consecutive_same(self, series):
        """Count how many times in a row we see the same value (stuck sensor detection)."""
        result = pd.Series(0, index=series.index, dtype=int)
        count = 0
        prev = None
        
        for i, val in enumerate(series.values):
            if pd.isna(val):
                count = 0
                prev = None
            elif val == prev:
                count += 1
            else:
                count = 1
                prev = val
            result.iloc[i] = count
        
        return result
    
    def _max_consecutive_same_window(self, series, window):
        """Max consecutive identical values in rolling window."""
        consecutive = self._count_consecutive_same(series)
        return consecutive.rolling(window, min_periods=1).max()
    
    def _compute_high_freq_energy(self, series, window):
        """Compute high-frequency energy to detect noisy sensors."""
        def high_freq(x):
            if len(x) < 4:
                return 0
            fft = np.fft.fft(x - x.mean())
            power = np.abs(fft) ** 2
            high_freq_power = power[len(power)//2:]
            return np.sum(high_freq_power) / (np.sum(power) + 1e-6)
        
        return series.rolling(window, min_periods=4).apply(high_freq, raw=True)
    
    def _compute_outlier_density(self, series, mean, std, window):
        """Count outliers (>3 sigma) in rolling window."""
        outliers = (np.abs(series - mean) > 3 * std).astype(int)
        return outliers.rolling(window, min_periods=1).sum()
    
    def _compute_autocorr(self, series, lag, window):
        """Compute rolling autocorrelation at given lag."""
        def autocorr(x):
            if len(x) < lag + 1:
                return 0
            x_centered = x - x.mean()
            c0 = np.sum(x_centered ** 2)
            if c0 == 0:
                return 0
            c_lag = np.sum(x_centered[:-lag] * x_centered[lag:])
            return c_lag / c0
        
        return series.rolling(window, min_periods=lag+1).apply(autocorr, raw=True)
    
    def _compute_trend_strength(self, series, window):
        """Compute trend strength using linear regression slope."""
        def trend(x):
            if len(x) < 3:
                return 0
            idx = np.arange(len(x))
            slope = np.polyfit(idx, x, 1)[0]
            return abs(slope)
        
        return series.rolling(window, min_periods=3).apply(trend, raw=True)
    
    def _compute_pattern_deviation(self, series, long_mean, long_std):
        """Measure deviation from long-term pattern."""
        return np.abs(series - long_mean) / (long_std + 1e-6)
    
    def get_feature_names(self):
        return [
            'oxygen_value', 'rolling_mean_60', 'rolling_std_60', 'rolling_median_60',
            'rolling_min_60', 'rolling_mean_360', 'rolling_std_360', 'rate_change_1min',
            'rate_change_5min', 'rate_change_60min', 'z_score_global', 'z_score_local',
            'dist_from_median', 'variance_30min', 'consecutive_same', 'range_30min',
            'coeff_variation', 'max_consecutive_same_60', 'value_change_rate',
            'point_to_point_variance', 'high_freq_energy', 'outlier_density_30min',
            'autocorr_lag5', 'trend_strength_60min', 'pattern_deviation'
        ]
    
    def save(self, filepath):
        """Save feature engineer to disk."""
        engineer_data = {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'fault_window': self.fault_window,
            'global_mean_': self.global_mean_,
            'global_std_': self.global_std_
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(engineer_data, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load feature engineer from disk."""
        engineer_data = joblib.load(filepath)
        engineer = cls(
            short_window=engineer_data['short_window'],
            long_window=engineer_data['long_window'],
            fault_window=engineer_data['fault_window']
        )
        engineer.global_mean_ = engineer_data['global_mean_']
        engineer.global_std_ = engineer_data['global_std_']
        return engineer
