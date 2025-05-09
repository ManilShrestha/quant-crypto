# Equity Pairs Trading Statistical Arbitrage

A sophisticated statistical arbitrage framework for equity markets that implements pairs trading strategies using cointegration analysis and mean reversion principles. This project focuses on identifying and trading correlated stock pairs while maintaining market neutrality.

## Features

- Automated pairs selection using cointegration tests
- Dynamic position sizing based on z-scores
- Real-time signal generation using mean reversion principles
- Comprehensive backtesting framework with transaction costs
- Risk management and portfolio optimization
- Interactive visualizations and performance analytics
- Detailed PDF report generation

## Project Structure

```
equity_stat_arb/
├── src/                   # Source code
│   ├── data/             # Data collection and processing
│   ├── analysis/         # Statistical analysis tools
│   ├── strategy/         # Trading strategy implementation
│   ├── backtest/         # Backtesting framework
│   └── visualization/    # Plotting and visualization tools
├── data/                 # Data storage
│   ├── raw/             # Raw price data
│   └── processed/       # Processed pairs data
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
├── config/              # Configuration files
├── reports/             # Generated reports and visualizations
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
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

## Usage

1. Data Collection:
```bash
python src/data/collect_data.py
```

2. Pairs Selection:
```bash
python src/analysis/select_pairs.py
```

3. Strategy Backtest:
```bash
python src/strategy/run_backtest.py
```

4. Generate Report:
```bash
python src/visualization/generate_report.py
```

## Trading Strategy

The strategy implements a sophisticated pairs trading approach:

1. **Pairs Selection**:
   - Cointegration testing using Engle-Granger method
   - Correlation analysis
   - Liquidity screening
   - Sector-based grouping

2. **Signal Generation**:
   - Z-score calculation for spread
   - Entry thresholds (typically ±2 standard deviations)
   - Exit thresholds (mean reversion)
   - Stop-loss and take-profit levels

3. **Position Management**:
   - Market-neutral portfolio construction
   - Dynamic position sizing
   - Risk parity allocation
   - Portfolio rebalancing

4. **Risk Management**:
   - Maximum position limits
   - Sector exposure limits
   - Drawdown controls
   - Volatility targeting

## Performance Metrics

- Total Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Average Trade Duration
- Beta to Market

## Technical Implementation

The project uses:
- pandas and numpy for data manipulation
- statsmodels for statistical analysis
- scipy for optimization
- matplotlib and seaborn for visualization
- yfinance for data collection
- reportlab for PDF generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 