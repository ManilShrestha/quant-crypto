# Crypto ML Statistical Arbitrage

A comprehensive framework for developing and evaluating trading strategies on the top 12 cryptocurrencies by trading volume. This project implements both machine learning and statistical arbitrage approaches for daily and intraday trading.

## Features

- Data collection from multiple exchanges using CCXT
- Feature engineering with technical indicators
- Machine learning models (XGBoost, LSTM)
- Statistical arbitrage strategies
- Backtesting framework
- Performance evaluation and visualization

## Project Structure

```
crypto_ml_stat_arb/
├── data/                  # Data storage
├── src/                   # Source code
│   ├── data/             # Data collection and processing
│   ├── features/         # Feature engineering
│   ├── models/           # ML models
│   ├── strategies/       # Trading strategies
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit tests
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
```

## Usage

1. Data Collection:
```python
from src.data.collector import DataCollector
collector = DataCollector()
data = collector.fetch_historical_data('BTC/USDT', '1h')
```

2. Feature Engineering:
```python
from src.features.technical import TechnicalFeatures
features = TechnicalFeatures(data)
processed_data = features.compute_all()
```

3. Strategy Backtesting:
```python
from src.strategies.ml_strategy import MLStrategy
strategy = MLStrategy(processed_data)
results = strategy.backtest()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 