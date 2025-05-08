# Crypto ML Statistical Arbitrage

A machine learning-based trading strategy framework for cryptocurrency markets. This project implements an XGBoost-based trading strategy that uses technical indicators to generate trading signals and includes a comprehensive backtesting framework.

## Features

- Data collection from Yahoo Finance
- Technical indicator calculation using pandas_ta
- Machine learning model (XGBoost) for signal generation
- Backtesting framework with realistic trading simulation
- Performance metrics calculation (returns, Sharpe ratio, drawdown)
- PDF report generation for strategy analysis

## Project Structure

```
stat_arb/
├── src/                   # Source code
│   ├── features/         # Technical indicators
│   └── strategies/       # ML trading strategy
├── data/                 # Data storage
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── example.py           # Example usage script
├── generate_report.py   # PDF report generator
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

1. Run the example script:
```bash
python example.py
```

This will:
- Fetch historical data for BTC/USD
- Calculate technical indicators
- Train the ML model
- Run a backtest
- Save results to CSV

2. Generate a detailed PDF report:
```bash
python generate_report.py
```

## Trading Strategy

The strategy uses an XGBoost classifier to predict price movements based on technical indicators:

1. **Signal Generation**:
   - Buy (1): When model predicts significant upward movement
   - Sell (-1): When model predicts significant downward movement
   - Hold (0): When model predicts neutral movement

2. **Position Management**:
   - Buy signals: Enter full position using available cash
   - Sell signals: Exit all positions
   - Hold signals: Maintain current position

3. **Performance Metrics**:
   - Total Return
   - Annual Return
   - Sharpe Ratio
   - Maximum Drawdown

## Technical Implementation

The project uses:
- pandas_ta for technical indicators
- XGBoost for machine learning
- pandas for data manipulation
- reportlab for PDF generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 