import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
from scipy import stats
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PairsSelector:
    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        min_correlation: float = 0.7,
        max_pvalue: float = 0.05
    ):
        """Initialize the pairs selector.
        
        Args:
            data_dir (str): Directory containing raw price data
            output_dir (str): Directory to store processed pairs data
            min_correlation (float): Minimum correlation threshold
            max_pvalue (float): Maximum p-value for cointegration test
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.min_correlation = min_correlation
        self.max_pvalue = max_pvalue
        os.makedirs(output_dir, exist_ok=True)
        
    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        """Load price data from CSV files.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of price DataFrames
        """
        price_data = {}
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                symbol = file.replace('.csv', '')
                df = pd.read_csv(os.path.join(self.data_dir, file), index_col=0, parse_dates=True)
                price_data[symbol] = df['Close']
        return price_data
    
    def calculate_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix for all pairs.
        
        Args:
            price_data (Dict[str, pd.Series]): Dictionary of price series
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        df = pd.DataFrame(price_data)
        return df.corr()
    
    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[float, float]:
        """Test for cointegration between two price series.
        
        Args:
            series1 (pd.Series): First price series
            series2 (pd.Series): Second price series
            
        Returns:
            Tuple[float, float]: (test statistic, p-value)
        """
        # Align the series
        series1, series2 = series1.align(series2, join='inner')
        
        # Perform cointegration test
        score, pvalue, _ = coint(series1, series2)
        return score, pvalue
    
    def find_cointegrated_pairs(self, price_data: Dict[str, pd.Series]) -> List[Dict]:
        """Find cointegrated pairs from price data.
        
        Args:
            price_data (Dict[str, pd.Series]): Dictionary of price series
            
        Returns:
            List[Dict]: List of cointegrated pairs with their statistics
        """
        symbols = list(price_data.keys())
        cointegrated_pairs = []
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(price_data)
        
        # Test each pair
        for i in tqdm(range(len(symbols)), desc="Testing pairs"):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Check correlation first
                correlation = corr_matrix.loc[symbol1, symbol2]
                if abs(correlation) < self.min_correlation:
                    continue
                
                # Test for cointegration
                score, pvalue = self.test_cointegration(
                    price_data[symbol1],
                    price_data[symbol2]
                )
                
                if pvalue < self.max_pvalue:
                    pair_info = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'cointegration_score': score,
                        'pvalue': pvalue
                    }
                    cointegrated_pairs.append(pair_info)
        
        return cointegrated_pairs
    
    def calculate_pair_statistics(self, pairs: List[Dict], price_data: Dict[str, pd.Series]) -> List[Dict]:
        """Calculate additional statistics for each pair.
        
        Args:
            pairs (List[Dict]): List of cointegrated pairs
            price_data (Dict[str, pd.Series]): Dictionary of price series
            
        Returns:
            List[Dict]: Pairs with additional statistics
        """
        for pair in pairs:
            symbol1, symbol2 = pair['symbol1'], pair['symbol2']
            series1, series2 = price_data[symbol1], price_data[symbol2]
            
            # Calculate spread
            spread = series1 - series2
            
            # Calculate spread statistics
            pair['spread_mean'] = spread.mean()
            pair['spread_std'] = spread.std()
            pair['spread_skew'] = stats.skew(spread)
            pair['spread_kurtosis'] = stats.kurtosis(spread)
            
            # Calculate trading statistics
            pair['avg_daily_volume1'] = price_data[symbol1].mean()
            pair['avg_daily_volume2'] = price_data[symbol2].mean()
            
        return pairs
    
    def save_pairs(self, pairs: List[Dict]):
        """Save selected pairs to JSON file.
        
        Args:
            pairs (List[Dict]): List of selected pairs
        """
        output_file = os.path.join(self.output_dir, 'selected_pairs.json')
        with open(output_file, 'w') as f:
            json.dump(pairs, f, indent=4)
        logger.info(f"Saved {len(pairs)} pairs to {output_file}")

def main():
    # Initialize pairs selector
    selector = PairsSelector()
    
    # Load price data
    price_data = selector.load_price_data()
    logger.info(f"Loaded price data for {len(price_data)} symbols")
    
    # Find cointegrated pairs
    pairs = selector.find_cointegrated_pairs(price_data)
    logger.info(f"Found {len(pairs)} cointegrated pairs")
    
    # Calculate additional statistics
    pairs = selector.calculate_pair_statistics(pairs, price_data)
    
    # Save results
    selector.save_pairs(pairs)

if __name__ == "__main__":
    main() 