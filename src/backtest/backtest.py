import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

class PairsBacktest:
    def __init__(
        self,
        data_dir: str = "data/raw",
        pairs_file: str = "data/processed/selected_pairs.json",
        initial_capital: float = 1000000.0,
        position_size: float = 0.02,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss: float = 3.0,
        transaction_cost: float = 0.001,
        lookback: int = 20,
        max_positions: int = 20,
        max_corr: float = 0.8
    ):
        """Initialize the backtesting framework.
        
        Args:
            data_dir (str): Directory containing price data
            pairs_file (str): Path to selected pairs JSON file
            initial_capital (float): Initial capital for backtest
            position_size (float): Position size as fraction of capital
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
            stop_loss (float): Stop loss threshold in standard deviations
            transaction_cost (float): Transaction cost as fraction of trade value
            lookback (int): Lookback period for z-score calculation
            max_positions (int): Maximum number of concurrent positions
            max_corr (float): Maximum allowed correlation between active pairs
        """
        self.data_dir = data_dir
        self.pairs_file = pairs_file
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.transaction_cost = transaction_cost
        self.lookback = lookback
        self.max_positions = max_positions
        self.max_corr = max_corr
        
        # Load pairs
        try:
            with open(pairs_file, 'r') as f:
                self.pairs = json.load(f)
        except FileNotFoundError:
            logger.error(f"Pairs file not found: {pairs_file}")
            self.pairs = []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in pairs file: {pairs_file}")
            self.pairs = []
            
        # Initialize results
        self.results = {
            'trades': [],
            'equity_curve': [],
            'positions': {},
            'metrics': {}
        }
        
        # Store spread histories
        self.spread_histories = {}
        
        # Store returns for correlation calculation
        self.returns_histories = {}
        
    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        """Load price data for all symbols.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of price DataFrames
        """
        price_data = {}
        try:
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            logger.info(f"Found {len(files)} price data files")
            
            for file in tqdm(files, desc="Loading price data"):
                try:
                    symbol = file.replace('.csv', '')
                    df = pd.read_csv(
                        os.path.join(self.data_dir, file),
                        index_col=0,
                        parse_dates=True,
                        date_format='ISO8601'
                    )
                    if not df.empty:
                        price_data[symbol] = df
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
                    continue
                    
            logger.info(f"Successfully loaded {len(price_data)} price data files")
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
        return price_data
    
    def calculate_zscore(self, spread_history: List[float]) -> float:
        """Calculate z-score of latest spread value.
        
        Args:
            spread_history (List[float]): Historical spread values
            
        Returns:
            float: Z-score of latest spread
        """
        if len(spread_history) < self.lookback:
            return 0.0
            
        lookback_spread = spread_history[-self.lookback:]
        mean = np.mean(lookback_spread)
        std = np.std(lookback_spread)
        
        if std == 0:
            return 0.0
            
        return (spread_history[-1] - mean) / std
    
    def calculate_correlation(self, symbol1: str, symbol2: str, price_data: Dict[str, pd.DataFrame], lookback: int = 60) -> float:
        """Calculate correlation between two symbols' returns.
        
        Args:
            symbol1 (str): First symbol
            symbol2 (str): Second symbol
            price_data (Dict[str, pd.DataFrame]): Price data dictionary
            lookback (int): Lookback period for correlation
            
        Returns:
            float: Correlation coefficient
        """
        try:
            # Get returns
            returns1 = price_data[symbol1]['Close'].pct_change()
            returns2 = price_data[symbol2]['Close'].pct_change()
            
            # Calculate correlation
            if len(returns1) > lookback and len(returns2) > lookback:
                corr = returns1[-lookback:].corr(returns2[-lookback:])
                return corr if not np.isnan(corr) else 0.0
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
        return 0.0
    
    def check_correlation_constraints(self, new_pair: Tuple[str, str], positions: Dict[str, Dict], price_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if adding a new pair would violate correlation constraints.
        
        Args:
            new_pair (Tuple[str, str]): New pair to check
            positions (Dict[str, Dict]): Current positions
            price_data (Dict[str, pd.DataFrame]): Price data dictionary
            
        Returns:
            bool: True if constraints are satisfied, False otherwise
        """
        new_symbol1, new_symbol2 = new_pair
        
        # Check correlation with existing positions
        for pos in positions.values():
            pos_symbol1, pos_symbol2 = pos['symbol1'], pos['symbol2']
            
            # Check correlation between all symbol combinations
            correlations = [
                abs(self.calculate_correlation(new_symbol1, pos_symbol1, price_data)),
                abs(self.calculate_correlation(new_symbol1, pos_symbol2, price_data)),
                abs(self.calculate_correlation(new_symbol2, pos_symbol1, price_data)),
                abs(self.calculate_correlation(new_symbol2, pos_symbol2, price_data))
            ]
            
            if any(c > self.max_corr for c in correlations):
                return False
        
        return True
    
    def run_backtest(self, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest for all pairs.
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Dictionary of price DataFrames
            
        Returns:
            Dict: Backtest results
        """
        if not price_data:
            logger.error("No price data available for backtest")
            return self.results
            
        if not self.pairs:
            logger.error("No pairs available for backtest")
            return self.results
            
        capital = self.initial_capital
        positions = {}
        equity_curve = []
        
        # Get all dates
        try:
            dates = sorted(list(set.intersection(
                *[set(df.index) for df in price_data.values()]
            )))
        except Exception as e:
            logger.error(f"Error getting common dates: {e}")
            return self.results
            
        if not dates:
            logger.error("No common dates found across price data")
            return self.results
        
        for date in tqdm(dates, desc="Running backtest"):
            daily_pnl = 0
            
            # Update existing positions
            for pair_id, position in list(positions.items()):
                try:
                    symbol1, symbol2 = position['symbol1'], position['symbol2']
                    price1 = price_data[symbol1].loc[date, 'Close']
                    price2 = price_data[symbol2].loc[date, 'Close']
                    
                    # Calculate P&L
                    pnl = (
                        position['size1'] * (price1 - position['entry_price1']) +
                        position['size2'] * (price2 - position['entry_price2'])
                    )
                    daily_pnl += pnl
                    
                    # Update spread history and calculate z-score
                    spread = price1 - price2
                    if pair_id not in self.spread_histories:
                        self.spread_histories[pair_id] = []
                    self.spread_histories[pair_id].append(spread)
                    zscore = self.calculate_zscore(self.spread_histories[pair_id])
                    
                    # Check stop loss
                    if abs(zscore) > self.stop_loss:
                        # Close position
                        daily_pnl -= (
                            abs(position['size1'] * price1) +
                            abs(position['size2'] * price2)
                        ) * self.transaction_cost
                        del positions[pair_id]
                        
                        # Record trade
                        self.results['trades'].append({
                            'pair_id': pair_id,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'pnl': pnl,
                            'exit_type': 'stop_loss'
                        })
                except Exception as e:
                    logger.error(f"Error updating position {pair_id}: {e}")
                    continue
            
            # Check for new positions if we have capacity
            if len(positions) < self.max_positions:
                for pair in self.pairs:
                    try:
                        symbol1, symbol2 = pair['symbol1'], pair['symbol2']
                        pair_id = f"{symbol1}_{symbol2}"
                        
                        if symbol1 not in price_data or symbol2 not in price_data:
                            continue
                            
                        if not self.check_correlation_constraints((symbol1, symbol2), positions, price_data):
                            continue
                            
                        price1 = price_data[symbol1].loc[date, 'Close']
                        price2 = price_data[symbol2].loc[date, 'Close']
                        
                        # Update spread history and calculate z-score
                        spread = price1 - price2
                        if pair_id not in self.spread_histories:
                            self.spread_histories[pair_id] = []
                        self.spread_histories[pair_id].append(spread)
                        zscore = self.calculate_zscore(self.spread_histories[pair_id])
                        
                        # Entry conditions
                        if pair_id not in positions:
                            # Calculate position size based on volatility
                            vol1 = price_data[symbol1]['Close'].pct_change().std()
                            vol2 = price_data[symbol2]['Close'].pct_change().std()
                            pair_vol = np.sqrt(vol1**2 + vol2**2)
                            
                            # Adjust position size inversely with volatility
                            adjusted_size = self.position_size / (pair_vol * np.sqrt(252))
                            size = min(adjusted_size, self.position_size) * capital / (price1 + price2)
                            
                            if zscore > self.entry_threshold:
                                # Short spread
                                positions[pair_id] = {
                                    'symbol1': symbol1,
                                    'symbol2': symbol2,
                                    'size1': -size,
                                    'size2': size,
                                    'entry_price1': price1,
                                    'entry_price2': price2,
                                    'entry_date': date,
                                    'entry_zscore': zscore
                                }
                                daily_pnl -= (
                                    abs(size * price1) +
                                    abs(size * price2)
                                ) * self.transaction_cost
                                
                            elif zscore < -self.entry_threshold:
                                # Long spread
                                positions[pair_id] = {
                                    'symbol1': symbol1,
                                    'symbol2': symbol2,
                                    'size1': size,
                                    'size2': -size,
                                    'entry_price1': price1,
                                    'entry_price2': price2,
                                    'entry_date': date,
                                    'entry_zscore': zscore
                                }
                                daily_pnl -= (
                                    abs(size * price1) +
                                    abs(size * price2)
                                ) * self.transaction_cost
                        
                        # Exit conditions
                        elif abs(zscore) < self.exit_threshold:
                            # Close position
                            position = positions[pair_id]
                            pnl = (
                                position['size1'] * (price1 - position['entry_price1']) +
                                position['size2'] * (price2 - position['entry_price2'])
                            )
                            daily_pnl += pnl
                            daily_pnl -= (
                                abs(position['size1'] * price1) +
                                abs(position['size2'] * price2)
                            ) * self.transaction_cost
                            
                            # Record trade
                            self.results['trades'].append({
                                'pair_id': pair_id,
                                'entry_date': position['entry_date'],
                                'exit_date': date,
                                'pnl': pnl,
                                'entry_zscore': position['entry_zscore'],
                                'exit_zscore': zscore,
                                'exit_type': 'target'
                            })
                            
                            del positions[pair_id]
                            
                        if len(positions) >= self.max_positions:
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing pair {pair_id}: {e}")
                        continue
            
            # Update capital and equity curve
            capital += daily_pnl
            equity_curve.append({
                'date': date,
                'capital': capital,
                'daily_pnl': daily_pnl,
                'active_positions': len(positions)
            })
        
        # Calculate performance metrics
        try:
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('date', inplace=True)
            
            returns = equity_df['capital'].pct_change()
            
            # Handle edge cases for metrics calculation
            total_return = (capital - self.initial_capital) / self.initial_capital
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
            max_drawdown = (equity_df['capital'] / equity_df['capital'].cummax() - 1).min()
            win_rate = len([t for t in self.results['trades'] if t['pnl'] > 0]) / len(self.results['trades']) if self.results['trades'] else 0
            profit_factor = abs(
                sum(t['pnl'] for t in self.results['trades'] if t['pnl'] > 0) /
                sum(t['pnl'] for t in self.results['trades'] if t['pnl'] < 0)
            ) if sum(t['pnl'] for t in self.results['trades'] if t['pnl'] < 0) != 0 else 0
            
            self.results['metrics'] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(self.results['trades'])
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            self.results['metrics'] = {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0
            }
        
        self.results['equity_curve'] = equity_curve
        self.results['positions'] = positions
        
        return self.results
    
    def save_results(self, output_file: str = "backtest_results.json"):
        """Save backtest results to JSON file.
        
        Args:
            output_file (str): Path to output file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=4, cls=DateTimeEncoder)
            logger.info(f"Saved backtest results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    # Initialize backtest
    backtest = PairsBacktest()
    
    # Load price data
    price_data = backtest.load_price_data()
    logger.info(f"Loaded price data for {len(price_data)} symbols")
    
    # Run backtest
    results = backtest.run_backtest(price_data)
    logger.info("Backtest completed")
    
    # Print metrics if available
    metrics = results.get('metrics', {})
    if metrics:
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
    else:
        logger.warning("No metrics available - backtest may have failed")
    
    # Save results
    backtest.save_results()

if __name__ == "__main__":
    main() 