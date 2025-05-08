import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Optional

class TechnicalFeatures:
    """
    A class for computing technical indicators and features from OHLCV data.
    """
    
    def __init__(self):
        """
        Initialize the technical features calculator.
        """
        pass
        
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all technical indicators
        """
        df = data.copy()
        
        try:
            # Moving Averages
            for window in [7, 14, 21, 50, 200]:
                df[f'sma_{window}'] = ta.sma(df['close'], length=window)
                df[f'ema_{window}'] = ta.ema(df['close'], length=window)
            
            # RSI
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # Stochastic Oscillator
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            
            # ATR
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Volume Indicators
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['volume_sma_20'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Returns and Price Changes
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change'] = df['close'].diff()
            df['price_change_pct'] = df['close'].pct_change()
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return pd.DataFrame() 