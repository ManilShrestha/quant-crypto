import os
from datetime import datetime, timedelta
from src.data.collector_factory import DataCollectorFactory
from src.features.technical import TechnicalFeatures
from src.strategies.ml_strategy import MLStrategy

def main():
    # Create data collector
    collector = DataCollectorFactory.create_collector('yahoo')
    
    # Print supported symbols and timeframes
    print("\nSupported Symbols:")
    for symbol in collector.get_supported_symbols().values():
        print(f"- {symbol}")
        
    print("\nSupported Timeframes:")
    for tf, desc in collector.get_supported_timeframes().items():
        print(f"- {tf}: {desc}")
    
    # Fetch historical data
    print("\nFetching historical data...")
    since = datetime.now() - timedelta(days=365)
    data = collector.fetch_historical_data('BTC/USD', timeframe='1h', since=since)
    
    if data.empty:
        print("Error: No data fetched")
        return
        
    print(f"\nFetched {len(data)} candles of BTC/USD data")
    print("\nSample data:")
    print(data.head())
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    features = TechnicalFeatures()
    indicators = features.calculate_all_indicators(data)
    
    print("\nSample indicators:")
    print(indicators.head())
    
    # Initialize and train ML strategy
    print("\nInitializing ML strategy...")
    strategy = MLStrategy(
        data=indicators,
        target_col='returns',
        prediction_threshold=0.02  # 2% threshold for trading signals
    )
    
    # Train the model
    print("\nTraining model...")
    strategy.train_model()
    
    # Run backtest
    print("\nRunning backtest...")
    initial_capital = 100000  # $100,000
    results = strategy.backtest(initial_capital=initial_capital)
    
    # Print backtest results
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Annual Return: {results['annual_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    
    # Save results
    results['portfolio'].to_csv('backtest_results.csv')
    print("\nResults saved to backtest_results.csv")

if __name__ == "__main__":
    main() 