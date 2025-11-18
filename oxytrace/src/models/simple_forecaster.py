"""
Oxygen forecaster with seasonal patterns and trend estimation
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class SimpleOxygenForecaster:
    
    def __init__(self, short_window=1440, long_window=10080, alpha=0.3):
        self.short_window = short_window
        self.long_window = long_window
        self.alpha = alpha
        
        self.global_mean_ = None
        self.global_std_ = None
        self.trend_ = None
        self.last_values_ = None
        self.last_timestamp_ = None
        self.daily_pattern_ = None
        self.weekly_pattern_ = None
        self.has_seasonality_ = False
        self.residual_std_ = None
    
    def fit(self, df, value_col='Oxygen[%sat]', time_col='time'):
        oxygen = df[value_col].dropna()
        
        self.global_mean_ = oxygen.mean()
        self.global_std_ = oxygen.std()
        
        if time_col in df.columns:
            df_clean = df[df[value_col].notna()].copy()
            df_clean['datetime'] = pd.to_datetime(df_clean[time_col])
            df_clean['minute_of_day'] = df_clean['datetime'].dt.hour * 60 + df_clean['datetime'].dt.minute
            df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek
            
            
            if len(df_clean) >= self.short_window * 7:
                daily_avg = df_clean.groupby('minute_of_day')[value_col].mean()
                self.daily_pattern_ = (daily_avg - self.global_mean_).values
                
                if len(self.daily_pattern_) < 1440:
                    full_pattern = np.zeros(1440)
                    full_pattern[daily_avg.index] = self.daily_pattern_
                    self.daily_pattern_ = full_pattern
                
                self.has_seasonality_ = True
            
            if len(df_clean) >= self.long_window * 2:
                weekly_avg = df_clean.groupby('day_of_week')[value_col].mean()
                self.weekly_pattern_ = (weekly_avg - self.global_mean_).values
            
            self.last_timestamp_ = df_clean['datetime'].iloc[-1]
        
        if len(oxygen) >= self.long_window:
            ewma = oxygen.ewm(alpha=self.alpha).mean()
            recent_ewma = ewma.iloc[-self.short_window:].mean()
            older_ewma = ewma.iloc[-self.long_window:-self.short_window].mean()
            self.trend_ = (recent_ewma - older_ewma) / self.short_window
        else:
            self.trend_ = 0
        
        if len(oxygen) >= self.short_window:
            self.last_values_ = oxygen.ewm(alpha=self.alpha).mean().iloc[-self.short_window:].values
        else:
            self.last_values_ = oxygen.values
        
        if self.has_seasonality_ and time_col in df.columns and len(df_clean) >= self.short_window:
            sample_size = min(10000, len(df_clean))
            sample_df = df_clean.iloc[-sample_size:].copy()
            
            predictions = np.zeros(len(sample_df))
            for i in range(len(sample_df)):
                pred = self.global_mean_
                if self.daily_pattern_ is not None:
                    minute = sample_df['minute_of_day'].iloc[i]
                    pred += self.daily_pattern_[minute]
                if self.weekly_pattern_ is not None:
                    day = sample_df['day_of_week'].iloc[i]
                    pred += self.weekly_pattern_[day] * 0.3
                predictions[i] = pred
            
            residuals = sample_df[value_col].values - predictions
            self.residual_std_ = np.std(residuals)
        else:
            self.residual_std_ = self.global_std_
    
    def predict(self, horizon=10080, return_uncertainty=True):
        if self.global_mean_ is None:
            raise RuntimeError("Must call fit() before predict()")
        
        if self.last_timestamp_ is not None:
            future_times = pd.date_range(
                start=self.last_timestamp_ + pd.Timedelta(minutes=1),
                periods=horizon,
                freq='1min'
            )
        else:
            future_times = None
        
        recent_mean = self.last_values_.mean()
        predictions = np.zeros(horizon)
        
        for t in range(horizon):
            pred = recent_mean + (self.trend_ * t)
            
            if self.has_seasonality_ and future_times is not None:
                if self.daily_pattern_ is not None:
                    minute_of_day = (future_times[t].hour * 60 + future_times[t].minute)
                    pred += self.daily_pattern_[minute_of_day]
                
                if self.weekly_pattern_ is not None:
                    day_of_week = future_times[t].dayofweek
                    pred += self.weekly_pattern_[day_of_week] * 0.3
            
            predictions[t] = pred
        
        result = pd.DataFrame({'predicted_oxygen': predictions})
        
        if return_uncertainty:
            base_std = self.residual_std_ if self.residual_std_ is not None else self.global_std_
            time_factor = np.sqrt(1 + np.arange(horizon) / self.short_window)
            time_factor = np.minimum(time_factor, 3.0)
            uncertainty = base_std * time_factor * 1.5
            
            result['lower_bound_95'] = predictions - 1.96 * uncertainty
            result['upper_bound_95'] = predictions + 1.96 * uncertainty
            result['uncertainty'] = uncertainty
        
        if future_times is not None:
            result.insert(0, 'time', future_times)
        
        return result
    
    def save(self, filepath):
        data = {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'alpha': self.alpha,
            'global_mean_': self.global_mean_,
            'global_std_': self.global_std_,
            'trend_': self.trend_,
            'last_values_': self.last_values_,
            'last_timestamp_': self.last_timestamp_,
            'daily_pattern_': self.daily_pattern_,
            'weekly_pattern_': self.weekly_pattern_,
            'has_seasonality_': self.has_seasonality_,
            'residual_std_': self.residual_std_
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, filepath)
    
    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        forecaster = cls(
            short_window=data['short_window'],
            long_window=data['long_window'],
            alpha=data.get('alpha', 0.3)
        )
        forecaster.global_mean_ = data['global_mean_']
        forecaster.global_std_ = data['global_std_']
        forecaster.trend_ = data['trend_']
        forecaster.last_values_ = data['last_values_']
        forecaster.last_timestamp_ = data['last_timestamp_']
        forecaster.daily_pattern_ = data.get('daily_pattern_')
        forecaster.weekly_pattern_ = data.get('weekly_pattern_')
        forecaster.has_seasonality_ = data.get('has_seasonality_', False)
        forecaster.residual_std_ = data.get('residual_std_')
        return forecaster
