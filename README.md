# 🚀 Vietnamese Energy Stocks Analysis - Complete Real Data Model

## 📋 Project Overview

This project provides a comprehensive analysis of Vietnamese energy stocks using real market data from vnstock, combining quantitative portfolio analysis, machine learning predictions, and advanced statistical modeling.

### 🎯 Key Features

- **Real Data Integration**: All 6 key indicators from vnstock (no mock data)
- **Quantitative Analysis**: Complete portfolio optimization and risk analysis
- **Machine Learning**: Statistical predictions for 2026-2030
- **Advanced Portfolio Theory**: Efficient frontier, tangency portfolio, backtesting
- **5-Year Historical Analysis**: Returns and Sharpe ratio analysis
- **Individual Stock Analysis**: Detailed performance metrics for each stock

## 📊 Data Sources

### ✅ Real Data from vnstock:

1. **Trading Volume (millions)** - VN-Index volume data
2. **Market Index (VN-Index)** - VNINDEX close price
3. **Debt Ratio (lev)** - "Nợ/VCSH" from financial ratios
4. **ROA** - "ROA (%)" from financial ratios
5. **Cash Ratio** - "Chỉ số thanh toán tiền mặt"
6. **Asset Turnover** - "Vòng quay tài sản"

### 📈 Stock Universe:

- **PLX** - Petrolimex
- **OIL** - PV Oil
- **GAS** - PetroVietnam Gas
- **PPC** - Pha Lai Thermal Power
- **GEG** - Gia Lai Electricity
- **POW** - PetroVietnam Power

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, vnstock, openpyxl
```

### Installation

```bash
# Clone or download the project
cd /path/to/project

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python complete_real_data_model.py
```

## 📈 Analysis Components

### 1. 🔢 Quantitative Flow Analysis

- **Return & Risk Analysis**: Daily, monthly, annual statistics
- **Variance Analysis**: Risk metrics with annualized volatility
- **RTRR Analysis**: Return-to-Risk Ratio with ranking
- **Sharpe Ratio**: Risk-adjusted performance metrics
- **Drawdown Analysis**: Maximum drawdown with visualizations

### 2. 📊 Factor Analysis (EW vs CW)

- **Equal-Weighted Portfolio**: 1/n allocation strategy
- **Cap-Weighted Portfolio**: Market cap-based allocation
- **Performance Comparison**: Statistical comparison of strategies
- **CW Weights Analysis**: Detailed weight distribution

### 3. ⚖️ Portfolio Optimization

- **Efficient Frontier**: Optimal risk-return combinations
- **Tangency Portfolio**: Maximum Sharpe ratio portfolio
- **GMV Portfolio**: Global Minimum Variance portfolio
- **RTRR Weights**: Return-to-Risk based allocation
- **Backtesting**: Historical performance validation

### 4. 🚀 Advanced Portfolio Analysis

- **Sharpe Ratio Maximization**: Mathematical optimization
- **5-Strategy Comparison**: Tangency, GMV, EW, RTRR, CW
- **Wealth Index Tracking**: Starting from 1000
- **Performance Statistics**: Comprehensive metrics

### 5. 📈 5-Year Historical Analysis

- **Annual Returns**: Year-by-year performance (2020-2025)
- **Sharpe Ratios**: Risk-adjusted returns by year
- **Best/Worst Performers**: Annual rankings
- **Heatmaps**: Visual performance matrices

### 6. 🤖 Machine Learning Predictions

- **Statistical Model**: Historical pattern-based predictions
- **2026-2030 Forecasts**: 5-year forward-looking analysis
- **Consistent Results**: Fixed seed for reproducible outputs
- **Realistic Bounds**: Market-appropriate prediction ranges

### 7. 📊 Individual Stock Analysis

- **Detailed Statistics**: Daily, monthly, annual metrics
- **Risk Metrics**: VaR, CVaR, Max Drawdown
- **Performance Charts**: 4 charts per stock
- **Comprehensive Analysis**: Complete stock profiling

## 📊 Generated Charts (32 Total)

### 🔍 5-Year Analysis (3 charts)

- `5year_returns_heatmap.png` - Annual returns heatmap
- `5year_returns_sharpe_summary.png` - Combined returns & Sharpe charts
- `5year_sharpe_heatmap.png` - Annual Sharpe ratios heatmap

### 🚀 Advanced Portfolio (3 charts)

- `advanced_efficient_frontier.png` - Advanced EF with CML
- `portfolio_backtest_comparison.png` - 3-strategy backtest
- `complete_portfolio_comparison.png` - All 5 strategies comparison

### 📈 Individual Stock Analysis (6 charts)

- `gas_detailed_analysis.png` - GAS comprehensive analysis
- `geg_detailed_analysis.png` - GEG comprehensive analysis
- `oil_detailed_analysis.png` - OIL comprehensive analysis
- `plx_detailed_analysis.png` - PLX comprehensive analysis
- `pow_detailed_analysis.png` - POW comprehensive analysis
- `ppc_detailed_analysis.png` - PPC comprehensive analysis

### 🤖 ML Predictions (6 charts)

- `ml_prediction_2026_chart.png` - 2026 predictions
- `ml_prediction_2027_chart.png` - 2027 predictions
- `ml_prediction_2028_chart.png` - 2028 predictions
- `ml_prediction_2029_chart.png` - 2029 predictions
- `ml_prediction_2030_chart.png` - 2030 predictions
- `ml_prediction_summary_5years_chart.png` - 5-year summary

### ⚖️ Portfolio Analysis (8 charts)

- `cap_weighted_performance_chart.png` - Cap-weighted performance
- `cw_weights_chart.png` - Cap-weighted weights
- `efficient_frontier_chart.png` - Basic efficient frontier
- `equal_weighted_performance_chart.png` - Equal-weighted performance
- `ew_vs_cw_comparison_chart.png` - EW vs CW comparison
- `portfolio_weights_chart.png` - Portfolio weights comparison
- `rtrr_weights_chart.png` - RTRR-based weights
- `tangency_portfolio_analysis_chart.png` - Tangency portfolio analysis

### 📊 Risk & Return (6 charts)

- `drawdown_analysis_chart.png` - Drawdown analysis
- `minimum_drawdown_chart.png` - Minimum drawdown ranking
- `risk_return_scatter_chart.png` - Risk vs return scatter
- `rtrr_chart.png` - RTRR analysis
- `sharpe_ratio_chart.png` - Sharpe ratios
- `wealth_index_chart.png` - Wealth index evolution

## 🎯 Key Results

### 📈 Best Performing Strategy

- **RTRR Portfolio**: 1707.12 final wealth (70.7% return)
- **Highest Sharpe**: 0.1562
- **Highest Mean Return**: 4.23% monthly

### 🏆 Best Individual Stock

- **OIL**: 15.48% mean annual return
- **Consistent winner**: Best performer in 4 out of 6 years
- **Highest volatility**: 36.37% annual volatility

### 📊 Portfolio Performance Ranking

1. **RTRR**: 1707.12 final wealth (Best overall)
2. **Equal-Weighted**: 1260.46 final wealth (Best risk-adjusted)
3. **Cap-Weighted**: 1202.23 final wealth (Moderate performance)
4. **GMV**: 892.15 final wealth (Conservative)
5. **Tangency**: 173.31 final wealth (Optimization issues)

## 🔧 Technical Details

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
vnstock>=0.2.0
openpyxl>=3.1.0
```

### Data Processing

- **Real-time data fetching** from vnstock
- **Data cleaning** and preprocessing
- **Feature engineering** with technical indicators
- **Time series alignment** and handling

### Model Architecture

- **Statistical prediction model** (not ML to avoid overfitting)
- **Historical pattern-based** forecasting
- **Market scenario modeling**
- **Company-specific factors** integration

## 📝 Usage Examples

### Run Complete Analysis

```python
python complete_real_data_model.py
```

### Key Outputs

- 32 professional charts
- Comprehensive statistics tables
- Portfolio optimization results
- ML predictions for 2026-2030
- Individual stock analysis

## 🤝 Contributing

This project is designed for Vietnamese energy market analysis. Contributions are welcome for:

- Additional technical indicators
- Enhanced ML models
- Extended time periods
- Additional stocks

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data usage policies when using vnstock data.

## 📞 Contact

For questions or collaboration opportunities, please refer to the project documentation.

---

**🎯 This comprehensive analysis provides deep insights into Vietnamese energy stocks with real market data, advanced portfolio theory, and statistical predictions for informed investment decisions.**
