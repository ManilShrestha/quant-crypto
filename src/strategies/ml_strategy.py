import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from typing import Tuple, Dict, Optional

class MLStrategy:
    """
    A class for implementing machine learning-based trading strategies.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = 'returns',
        prediction_threshold: float = 0.02,
        train_size: float = 0.7,
        val_size: float = 0.15
    ):
        """
        Initialize the ML strategy.
        
        Args:
            data (pd.DataFrame): DataFrame with features and target
            target_col (str): Column name for target variable
            prediction_threshold (float): Threshold for trading signals
            train_size (float): Proportion of data for training
            val_size (float): Proportion of data for validation
        """
        self.data = data.copy()
        self.target_col = target_col
        self.prediction_threshold = prediction_threshold
        self.train_size = train_size
        self.val_size = val_size
        
        # Create target variable (0 for negative, 1 for neutral, 2 for positive)
        self.data['target'] = np.where(
            self.data[target_col] > prediction_threshold, 2,
            np.where(self.data[target_col] < -prediction_threshold, 0, 1)
        )
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Select features (exclude target and price columns)
        feature_cols = [col for col in self.data.columns 
                       if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = self.data[feature_cols]
        y = self.data['target']
        
        return X, y
        
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def train_model(self) -> None:
        """
        Train the XGBoost model.
        """
        X, y = self.prepare_features()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42,
            eval_metric=['mlogloss', 'merror']
        )
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val_scaled)
        print("\nValidation Set Performance:")
        print(f"Accuracy: {accuracy_score(y_val, val_pred):.3f}")
        print(f"Precision: {precision_score(y_val, val_pred, average='weighted'):.3f}")
        print(f"Recall: {recall_score(y_val, val_pred, average='weighted'):.3f}")
        print(f"F1 Score: {f1_score(y_val, val_pred, average='weighted'):.3f}")
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions.
        
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating signals")
            
        X, _ = self.prepare_features()
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Convert predictions back to trading signals (0 -> -1, 1 -> 0, 2 -> 1)
        signals = np.where(predictions == 0, -1, np.where(predictions == 1, 0, 1))
        
        # Add signals to data
        self.data['signal'] = signals
        
        return self.data
        
    def backtest(self, initial_capital: float = 100000.0) -> Dict:
        """
        Backtest the strategy.
        
        Args:
            initial_capital (float): Initial capital for backtesting
            
        Returns:
            Dict: Backtest results
        """
        if 'signal' not in self.data.columns:
            self.generate_signals()
            
        # Initialize portfolio
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['capital'] = pd.Series(initial_capital, index=self.data.index, dtype='float64')
        portfolio['position'] = pd.Series(0, index=self.data.index, dtype='float64')
        portfolio['holdings'] = pd.Series(0, index=self.data.index, dtype='float64')
        portfolio['cash'] = pd.Series(initial_capital, index=self.data.index, dtype='float64')
        
        # Simulate trading
        for i in range(1, len(portfolio)):
            signal = self.data['signal'].iloc[i]
            price = self.data['close'].iloc[i]
            
            # Update position based on signal
            if signal == 1:  # Buy signal
                portfolio.loc[portfolio.index[i], 'position'] = portfolio['cash'].iloc[i-1] / price
            elif signal == -1:  # Sell signal
                portfolio.loc[portfolio.index[i], 'position'] = 0
            else:  # Hold
                portfolio.loc[portfolio.index[i], 'position'] = portfolio['position'].iloc[i-1]
                
            # Update holdings and cash
            portfolio.loc[portfolio.index[i], 'holdings'] = portfolio['position'].iloc[i] * price
            portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - (
                portfolio['position'].iloc[i] - portfolio['position'].iloc[i-1]
            ) * price
            
            # Update total capital
            portfolio.loc[portfolio.index[i], 'capital'] = portfolio['holdings'].iloc[i] + portfolio['cash'].iloc[i]
            
        # Calculate returns
        portfolio['returns'] = portfolio['capital'].pct_change()
        
        # Calculate performance metrics
        total_return = (portfolio['capital'].iloc[-1] / initial_capital - 1) * 100
        annual_return = total_return * (252 / len(portfolio))
        sharpe_ratio = np.sqrt(252) * portfolio['returns'].mean() / portfolio['returns'].std()
        max_drawdown = (portfolio['capital'] / portfolio['capital'].cummax() - 1).min() * 100
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio': portfolio
        } 