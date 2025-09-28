# -*- coding: utf-8 -*-
"""
QUANTITATIVE FLOW ANALYSIS - VIETNAMESE ENERGY FIRMS
Phân tích định lượng theo flow từng bước với table data + chart

Flow:
+ Tính toán lợi nhuận
+ Ước lượng độ biến động  
+ Sharpe Ratio & Risk-Adjusted Metrics
+ Comparing Equal-Weighted (EW) vs Capitalization-Weighted (CW) Portfolios
+ Xác định trọng số danh mục
+ Danh mục tối đa hóa Sharpe Ratio
"""

# Import thư viện
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import vnstock để lấy dữ liệu Việt Nam
try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
    print("✅ vnstock imported successfully")
except ImportError:
    VNSTOCK_AVAILABLE = False
    print("⚠️ vnstock not available - will use Yahoo Finance only")

# Cấu hình plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Đồng nhất màu chủ đạo
PRIMARY_COLOR = '#060270'
SECONDARY_COLOR = '#0A0A8C'
ACCENT_COLOR = '#1A1A9C'
LIGHT_COLOR = '#2A2AAC'
DARK_COLOR = '#040260'

# Color palette cho charts đơn sắc
SINGLE_COLOR = PRIMARY_COLOR

# Color palette cho charts nhiều đường - sử dụng nhiều màu sắc
MULTI_COLORS = [
    '#060270',  # Primary blue
    '#FF6B6B',  # Coral red
    '#4ECDC4',  # Teal
    '#45B7D1',  # Sky blue
    '#96CEB4',  # Mint green
    '#FFEAA7',  # Yellow
    '#DDA0DD',  # Plum
    '#98D8C8',  # Seafoam
    '#F7DC6F',  # Light yellow
    '#BB8FCE',  # Light purple
    '#85C1E9',  # Light blue
    '#F8C471'   # Light orange
]

# Set default palette
sns.set_palette(MULTI_COLORS)

print("📊 QUANTITATIVE FLOW ANALYSIS - VIETNAMESE ENERGY FIRMS")
print("=" * 60)

"""
DATA COLLECTION - Thu thập dữ liệu
"""

# Danh sách 6 mã cổ phiếu năng lượng Việt Nam
tickers = ['PLX', 'OIL', 'GAS', 'PPC', 'GEG', 'POW']
company_names = {
    'PLX': 'Petrolimex',
    'OIL': 'PVOIL', 
    'GAS': 'Tổng Công ty Dầu Khí Việt Nam',
    'PPC': 'Nhiệt điện Phả Lại',
    'GEG': 'CTCP Điện Gia Lai',
    'POW': 'PV Power'
}

print(f"\n📈 Thu thập dữ liệu cho {len(tickers)} mã cổ phiếu năng lượng:")
for ticker in tickers:
    print(f"   {ticker}: {company_names[ticker]}")

# Thời gian dữ liệu
start_date = '2020-01-01'
end_date = '2025-08-15'

print(f"\n⏰ Giai đoạn dữ liệu: {start_date} đến {end_date}")

# Thu thập dữ liệu
all_data = {}
successful_tickers = []

def get_data_vnstock(ticker, start_date, end_date):
    """Lấy dữ liệu từ vnstock"""
    try:
        if not VNSTOCK_AVAILABLE:
            return None
        
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        data = stock.quote.history(symbol=ticker, start=start_date, end=end_date, interval='1D')
        
        if data is not None and len(data) > 0:
            data = data.rename(columns={
                'time': 'Date',
                'open': 'Open', 
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            return data
        return None
    except Exception as e:
        print(f"   ❌ vnstock error for {ticker}: {str(e)[:50]}...")
        return None
    """Lấy dữ liệu từ Yahoo Finance"""
    try:
        yahoo_ticker = f"{ticker}.VN"
        data = yf.download(yahoo_ticker, start=start_date, end=end_date, progress=False)
        
        if data is not None and len(data) > 0:
            return data
        return None
    except Exception as e:
        print(f"   ❌ Yahoo Finance error for {ticker}: {str(e)[:50]}...")
        return None

# Thu thập dữ liệu cho từng mã
for ticker in tickers:
    print(f"\n📊 Đang tải dữ liệu cho {ticker} ({company_names[ticker]})...")
    
    data = get_data_vnstock(ticker, start_date, end_date)
    
    if data is not None and len(data) > 0:
        all_data[ticker] = data
        successful_tickers.append(ticker)
        print(f"   ✅ {ticker}: {len(data)} ngày dữ liệu")
    else:
        print(f"   ❌ {ticker}: Không có dữ liệu")

print(f"\n📊 Tổng kết thu thập dữ liệu:")
print(f"   ✅ Thành công: {len(successful_tickers)}/{len(tickers)} mã")
print(f"   📈 Danh sách: {', '.join(successful_tickers)}")

if len(successful_tickers) == 0:
    print("❌ Không có dữ liệu nào! Thoát chương trình.")
    exit()

# Tạo DataFrame tổng hợp giá đóng cửa
price_data = pd.DataFrame()

for ticker in successful_tickers:
    data = all_data[ticker]
    price_data[ticker] = data['Close']

# Drop NaN values
price_data = price_data.dropna()

# Tính daily returns
df_return = price_data.pct_change().dropna()

print(f"\n📊 Kích thước dữ liệu sau preprocessing:")
print(f"   - Giá: {price_data.shape}")
print(f"   - Returns: {df_return.shape}")

"""
STEP 1: TÍNH TOÁN LỢI NHUẬN
"""

print(f"\n" + "="*60)
print(f"📈 STEP 1: TÍNH TOÁN LỢI NHUẬN")
print(f"="*60)

# Tính toán lợi nhuận
print(f"\n📊 Tính toán lợi nhuận:")

# Lợi nhuận trung bình hàng ngày
mean_values = df_return.mean()

# Hàng tháng (21 trading days)
monthly_mean = (1 + mean_values)**21 - 1

# Hàng năm (250 trading days)
annualized_mean = (1 + mean_values)**250 - 1

# Lợi nhuận tích lũy
compound_return = (df_return + 1).prod() - 1

# Tạo bảng tổng hợp lợi nhuận
summary_returns = pd.DataFrame({
    'Daily Mean': mean_values,
    'Monthly Mean': monthly_mean,
    'Annualized Mean': annualized_mean,
    'Cumulative Return': compound_return
})

print("📊 TABLE 1: SUMMARY RETURNS")
print("-" * 80)
print(summary_returns.round(6))

# Tạo chart cho lợi nhuận
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Daily Mean Returns
axes[0,0].bar(summary_returns.index, summary_returns['Daily Mean'], color=SINGLE_COLOR, alpha=0.7)
axes[0,0].set_title('Daily Mean Returns', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Daily Return')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

# 2. Annualized Returns
axes[0,1].bar(summary_returns.index, summary_returns['Annualized Mean'], color=SINGLE_COLOR, alpha=0.7)
axes[0,1].set_title('Annualized Mean Returns', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Annualized Return')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# 3. Cumulative Returns
colors = [ACCENT_COLOR if x > 0 else LIGHT_COLOR for x in summary_returns['Cumulative Return']]
axes[1,0].bar(summary_returns.index, summary_returns['Cumulative Return'], color=colors, alpha=0.7)
axes[1,0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Cumulative Return')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 4. Monthly vs Annualized Comparison
x_pos = np.arange(len(summary_returns.index))
width = 0.35
axes[1,1].bar(x_pos - width/2, summary_returns['Monthly Mean'], width, label='Monthly', alpha=0.7, color=SINGLE_COLOR)
axes[1,1].bar(x_pos + width/2, summary_returns['Annualized Mean'], width, label='Annualized', alpha=0.7, color=SINGLE_COLOR)
axes[1,1].set_title('Monthly vs Annualized Returns', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Return')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(summary_returns.index, rotation=45)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step1_returns_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

"""
STEP 2: ƯỚC LƯỢNG ĐỘ BIẾN ĐỘNG
"""

print(f"\n" + "="*60)
print(f"📊 STEP 2: ƯỚC LƯỢNG ĐỘ BIẾN ĐỘNG")
print(f"="*60)

print(f"\n📊 Ước lượng độ biến động:")

# Tính độ biến động
std_daily = df_return.std()
variance = df_return.var()
volatility = variance.pow(0.5)

# Semi-deviation (downside risk)
semideviation = df_return[df_return < 0].std(ddof=0)

# Annualized volatility
annualized_volatility = df_return.std() * np.sqrt(250)

# Tạo bảng tổng hợp rủi ro
risk_df = pd.DataFrame({
    'Daily Std': std_daily,
    'Variance': variance,
    'Volatility': volatility,
    'Semi-Deviation': semideviation,
    'Annualized Volatility': annualized_volatility
})

print("📊 TABLE 2: RISK METRICS")
print("-" * 80)
print(risk_df.round(6))

# Tạo chart cho độ biến động
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Daily Standard Deviation
axes[0,0].bar(risk_df.index, risk_df['Daily Std'], color=SINGLE_COLOR, alpha=0.7)
axes[0,0].set_title('Daily Standard Deviation', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Daily Std')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

# 2. Annualized Volatility
axes[0,1].bar(risk_df.index, risk_df['Annualized Volatility'], color=SINGLE_COLOR, alpha=0.7)
axes[0,1].set_title('Annualized Volatility', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Annualized Volatility')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# 3. Semi-Deviation (Downside Risk)
axes[1,0].bar(risk_df.index, risk_df['Semi-Deviation'], color=SINGLE_COLOR, alpha=0.7)
axes[1,0].set_title('Semi-Deviation (Downside Risk)', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Semi-Deviation')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# 4. Volatility vs Semi-Deviation Comparison
x_pos = np.arange(len(risk_df.index))
width = 0.35
axes[1,1].bar(x_pos - width/2, risk_df['Volatility'], width, label='Volatility', alpha=0.7, color=SINGLE_COLOR)
axes[1,1].bar(x_pos + width/2, risk_df['Semi-Deviation'], width, label='Semi-Deviation', alpha=0.7, color=SINGLE_COLOR)
axes[1,1].set_title('Volatility vs Semi-Deviation', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Risk Measure')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(risk_df.index, rotation=45)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step2_volatility_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

"""
STEP 3: SHARPE RATIO & RISK-ADJUSTED METRICS
"""

print(f"\n" + "="*60)
print(f"⚖️ STEP 3: SHARPE RATIO & RISK-ADJUSTED METRICS")
print(f"="*60)

# Risk-free rate
riskfree_rate = 0.027

# Annualized return from cumulative
n_days = len(df_return)
annualized_return = (df_return + 1).prod()**(250 / n_days) - 1

# Excess return
excess_return = annualized_return - riskfree_rate

# Sharpe Ratio
sharpe_ratio = excess_return / annualized_volatility
sharpe_ratio = sharpe_ratio.sort_values(ascending=False)

# Return to Risk Ratio (RTRR)
rtrr = annualized_return / annualized_volatility

print(f"\n📊 Risk-free rate: {riskfree_rate:.1%}")

print("📊 TABLE 3: RISK-ADJUSTED METRICS")
print("-" * 80)
risk_adjusted_df = pd.DataFrame({
    'Annualized Return': annualized_return,
    'Annualized Volatility': annualized_volatility,
    'Excess Return': excess_return,
    'Sharpe Ratio': sharpe_ratio,
    'RTRR': rtrr
})
print(risk_adjusted_df.round(6))

# Tạo chart cho risk-adjusted metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Sharpe Ratios
colors = [MULTI_COLORS[0] if x > 0 else MULTI_COLORS[1] for x in sharpe_ratio]
axes[0,0].bar(sharpe_ratio.index, sharpe_ratio.values, color=colors, alpha=0.7)
axes[0,0].set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Sharpe Ratio')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)
axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 2. Risk vs Return Scatter
axes[0,1].scatter(annualized_volatility, annualized_return, s=100, alpha=0.7, color=SINGLE_COLOR)
for i, ticker in enumerate(annualized_volatility.index):
    axes[0,1].annotate(ticker, (annualized_volatility[ticker], annualized_return[ticker]), 
                       xytext=(5, 5), textcoords='offset points')
axes[0,1].set_title('Risk vs Return Analysis', fontweight='bold')
axes[0,1].set_xlabel('Volatility')
axes[0,1].set_ylabel('Expected Return')
axes[0,1].grid(True, alpha=0.3)

# 3. Excess Returns
colors = [ACCENT_COLOR if x > 0 else LIGHT_COLOR for x in excess_return]
axes[1,0].bar(excess_return.index, excess_return.values, color=colors, alpha=0.7)
axes[1,0].set_title('Excess Returns (vs Risk-free Rate)', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Excess Return')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 4. RTRR vs Sharpe Ratio
axes[1,1].scatter(rtrr, sharpe_ratio, s=100, alpha=0.7, color=SINGLE_COLOR)
for i, ticker in enumerate(rtrr.index):
    axes[1,1].annotate(ticker, (rtrr[ticker], sharpe_ratio[ticker]), 
                       xytext=(5, 5), textcoords='offset points')
axes[1,1].set_title('RTRR vs Sharpe Ratio', fontweight='bold')
axes[1,1].set_xlabel('Return to Risk Ratio')
axes[1,1].set_ylabel('Sharpe Ratio')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step3_risk_adjusted_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

"""
STEP 4: COMPARING EW vs CW PORTFOLIOS
"""

print(f"\n" + "="*60)
print(f"🔍 STEP 4: COMPARING EW vs CW PORTFOLIOS")
print(f"="*60)

print(f"\n📊 Comparing Equal-Weighted (EW) vs Capitalization-Weighted (CW) Portfolios")

# Equal-Weighted Portfolio
ew_weights = np.repeat(1 / len(successful_tickers), len(successful_tickers))
ew_return = df_return @ ew_weights

# Capitalization-Weighted Portfolio (sử dụng latest market cap proxy)
latest_prices = price_data.iloc[-1]
cw_weights = latest_prices / latest_prices.sum()
cw_return = df_return @ cw_weights

print(f"\n📈 Portfolio Weights:")
print(f"Equal-Weighted: {dict(zip(successful_tickers, ew_weights.round(4)))}")
print(f"Cap-Weighted: {dict(zip(successful_tickers, cw_weights.round(4)))}")

# So sánh performance
def perf_stats(r):
    return pd.Series({
        "Mean": r.mean(),
        "Volatility": r.std(),
        "Sharpe": (r.mean() - riskfree_rate/250) / r.std() if r.std() > 0 else 0
    })

portfolio_comparison = pd.DataFrame({
    "Equal-Weighted": perf_stats(ew_return),
    "Cap-Weighted": perf_stats(cw_return)
})

print("📊 TABLE 4: PORTFOLIO PERFORMANCE COMPARISON")
print("-" * 80)
print(portfolio_comparison.round(6))

# Tạo chart riêng cho Equal-Weighted Portfolio
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Equal-Weighted Portfolio Weights
x_pos = np.arange(len(successful_tickers))
axes[0,0].bar(x_pos, ew_weights, alpha=0.7, color=SINGLE_COLOR)
axes[0,0].set_title('Equal-Weighted Portfolio Weights', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Weight')
axes[0,0].set_xticks(x_pos)
axes[0,0].set_xticklabels(successful_tickers, rotation=45)
axes[0,0].grid(True, alpha=0.3)

# 2. Equal-Weighted Cumulative Performance
(1 + ew_return).cumprod().plot(ax=axes[0,1], linewidth=2, color=MULTI_COLORS[0])
axes[0,1].set_title('Equal-Weighted Portfolio: Cumulative Performance', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Cumulative Return')
axes[0,1].grid(True, alpha=0.3)

# 3. Equal-Weighted Performance Metrics
metrics = ['Mean', 'Volatility', 'Sharpe']
ew_values = [portfolio_comparison.loc[metric, 'Equal-Weighted'] for metric in metrics]

x_pos = np.arange(len(metrics))
axes[1,0].bar(x_pos, ew_values, alpha=0.7, color=SINGLE_COLOR)
axes[1,0].set_title('Equal-Weighted Portfolio: Performance Metrics', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Value')
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(metrics)
axes[1,0].grid(True, alpha=0.3)

# 4. Equal-Weighted Risk-Return
axes[1,1].scatter(portfolio_comparison.loc['Volatility', 'Equal-Weighted'], 
                  portfolio_comparison.loc['Mean', 'Equal-Weighted'], 
                  s=200, alpha=0.7, color=SINGLE_COLOR)
axes[1,1].set_title('Equal-Weighted Portfolio: Risk vs Return', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Volatility')
axes[1,1].set_ylabel('Mean Return')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step4a_equal_weighted_portfolio.png', dpi=300, bbox_inches='tight')
plt.show()

# Tạo chart riêng cho Capitalization-Weighted Portfolio
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Capitalization-Weighted Portfolio Weights
x_pos = np.arange(len(successful_tickers))
axes[0,0].bar(x_pos, cw_weights, alpha=0.7, color=SINGLE_COLOR)
axes[0,0].set_title('Capitalization-Weighted Portfolio Weights', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Weight')
axes[0,0].set_xticks(x_pos)
axes[0,0].set_xticklabels(successful_tickers, rotation=45)
axes[0,0].grid(True, alpha=0.3)

# 2. Capitalization-Weighted Cumulative Performance
(1 + cw_return).cumprod().plot(ax=axes[0,1], linewidth=2, color=MULTI_COLORS[1])
axes[0,1].set_title('Capitalization-Weighted Portfolio: Cumulative Performance', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Cumulative Return')
axes[0,1].grid(True, alpha=0.3)

# 3. Capitalization-Weighted Performance Metrics
metrics = ['Mean', 'Volatility', 'Sharpe']
cw_values = [portfolio_comparison.loc[metric, 'Cap-Weighted'] for metric in metrics]

x_pos = np.arange(len(metrics))
axes[1,0].bar(x_pos, cw_values, alpha=0.7, color=SINGLE_COLOR)
axes[1,0].set_title('Capitalization-Weighted Portfolio: Performance Metrics', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Value')
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(metrics)
axes[1,0].grid(True, alpha=0.3)

# 4. Capitalization-Weighted Risk-Return
axes[1,1].scatter(portfolio_comparison.loc['Volatility', 'Cap-Weighted'], 
                  portfolio_comparison.loc['Mean', 'Cap-Weighted'], 
                  s=200, alpha=0.7, color=SINGLE_COLOR)
axes[1,1].set_title('Capitalization-Weighted Portfolio: Risk vs Return', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Volatility')
axes[1,1].set_ylabel('Mean Return')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step4b_capitalization_weighted_portfolio.png', dpi=300, bbox_inches='tight')
plt.show()

"""
STEP 5: XÁC ĐỊNH TRỌNG SỐ DANH MỤC
"""

print(f"\n" + "="*60)
print(f"⚖️ STEP 5: XÁC ĐỊNH TRỌNG SỐ DANH MỤC")
print(f"="*60)

# Select top 3 stocks based on Sharpe ratio for portfolio optimization
top_3_stocks = sharpe_ratio.head(3).index.tolist()
print(f"\n🏆 Top 3 stocks selected for portfolio optimization: {top_3_stocks}")

# Filter data for top 3 stocks
top3_returns = df_return[top_3_stocks]
top3_expected_returns = annualized_return[top_3_stocks]
top3_cov_matrix = top3_returns.cov() * 250  # Annualized covariance

print(f"\n📊 Top 3 Stocks Analysis:")
top3_analysis = pd.DataFrame({
    'Expected Return': top3_expected_returns,
    'Volatility': annualized_volatility[top_3_stocks],
    'Sharpe Ratio': sharpe_ratio[top_3_stocks]
})

print("📊 TABLE 5: TOP 3 STOCKS ANALYSIS")
print("-" * 80)
print(top3_analysis.round(6))

# Portfolio optimization functions
def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.027):
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol if vol > 0 else 0

# Tạo chart cho top 3 stocks analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Top 3 Stocks Expected Returns
axes[0,0].bar(top3_analysis.index, top3_analysis['Expected Return'], color=SINGLE_COLOR, alpha=0.7)
axes[0,0].set_title('Top 3 Stocks: Expected Returns', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Expected Return')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

# 2. Top 3 Stocks Volatility
axes[0,1].bar(top3_analysis.index, top3_analysis['Volatility'], color=SINGLE_COLOR, alpha=0.7)
axes[0,1].set_title('Top 3 Stocks: Volatility', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Volatility')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# 3. Top 3 Stocks Sharpe Ratios
colors = [ACCENT_COLOR if x > 0 else LIGHT_COLOR for x in top3_analysis['Sharpe Ratio']]
axes[1,0].bar(top3_analysis.index, top3_analysis['Sharpe Ratio'], color=colors, alpha=0.7)
axes[1,0].set_title('Top 3 Stocks: Sharpe Ratios', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Sharpe Ratio')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 4. Risk-Return Scatter for Top 3
axes[1,1].scatter(top3_analysis['Volatility'], top3_analysis['Expected Return'], s=100, alpha=0.7, color=SINGLE_COLOR)
for i, ticker in enumerate(top3_analysis.index):
    axes[1,1].annotate(ticker, (top3_analysis['Volatility'][ticker], top3_analysis['Expected Return'][ticker]), 
                       xytext=(5, 5), textcoords='offset points')
axes[1,1].set_title('Top 3 Stocks: Risk vs Return', fontweight='bold')
axes[1,1].set_xlabel('Volatility')
axes[1,1].set_ylabel('Expected Return')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step5_portfolio_weights_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

"""
STEP 6: DANH MỤC TỐI ĐA HÓA SHARPE RATIO
"""

print(f"\n" + "="*60)
print(f"🏆 STEP 6: DANH MỤC TỐI ĐA HÓA SHARPE RATIO")
print(f"="*60)

print(f"\n🏆 Danh mục tối đa hóa Sharpe Ratio:")

# Danh mục tối đa hóa Sharpe Ratio
def optimize_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.027):
    """Tối ưu hóa portfolio để maximize Sharpe ratio"""
    n_assets = len(expected_returns)
    
    def objective(weights):
        return -portfolio_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple([(0, 1) for _ in range(n_assets)])
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

# Global Minimum Variance Portfolio
def optimize_min_variance(expected_returns, cov_matrix):
    """Tối ưu hóa portfolio để minimize variance"""
    n_assets = len(expected_returns)
    
    def objective(weights):
        return portfolio_volatility(weights, cov_matrix)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple([(0, 1) for _ in range(n_assets)])
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

# Calculate optimal portfolios for top 3 stocks
tangency_weights = optimize_sharpe_ratio(top3_expected_returns, top3_cov_matrix)
tangency_return = portfolio_return(tangency_weights, top3_expected_returns)
tangency_vol = portfolio_volatility(tangency_weights, top3_cov_matrix)
tangency_sharpe = portfolio_sharpe_ratio(tangency_weights, top3_expected_returns, top3_cov_matrix)

gmv_weights = optimize_min_variance(top3_expected_returns, top3_cov_matrix)
gmv_return = portfolio_return(gmv_weights, top3_expected_returns)
gmv_vol = portfolio_volatility(gmv_weights, top3_cov_matrix)
gmv_sharpe = portfolio_sharpe_ratio(gmv_weights, top3_expected_returns, top3_cov_matrix)

ew3_weights = np.array([1/3] * 3)
ew3_return = portfolio_return(ew3_weights, top3_expected_returns)
ew3_vol = portfolio_volatility(ew3_weights, top3_cov_matrix)
ew3_sharpe = portfolio_sharpe_ratio(ew3_weights, top3_expected_returns, top3_cov_matrix)

# Portfolio Results
portfolio_results = pd.DataFrame({
    'Tangency Portfolio': tangency_weights,
    'GMV Portfolio': gmv_weights,
    'Equal Weight (Top3)': ew3_weights
}, index=top_3_stocks)

portfolio_performance = pd.DataFrame({
    'Expected Return': [tangency_return, gmv_return, ew3_return],
    'Volatility': [tangency_vol, gmv_vol, ew3_vol],
    'Sharpe Ratio': [tangency_sharpe, gmv_sharpe, ew3_sharpe]
}, index=['Tangency Portfolio', 'GMV Portfolio', 'Equal Weight (Top3)'])

print("📊 TABLE 6A: PORTFOLIO WEIGHTS (TOP 3 STOCKS)")
print("-" * 80)
print(portfolio_results.round(6))

print("\n📊 TABLE 6B: PORTFOLIO PERFORMANCE (TOP 3 STOCKS)")
print("-" * 80)
print(portfolio_performance.round(6))

# Tạo chart riêng cho Tangency Portfolio
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Tangency Portfolio Weights
tangency_weights_df = pd.DataFrame({
    'Tangency Portfolio': tangency_weights
}, index=top_3_stocks)
tangency_weights_df.plot(kind='bar', ax=axes[0,0], width=0.6, color=SINGLE_COLOR, alpha=0.7)
axes[0,0].set_title('Tangency Portfolio Weights (Top 3 Stocks)', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Weight')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

# 2. Tangency Portfolio Performance Metrics
tangency_metrics = ['Expected Return', 'Volatility', 'Sharpe Ratio']
tangency_values = [tangency_return, tangency_vol, tangency_sharpe]
axes[0,1].bar(tangency_metrics, tangency_values, color=SINGLE_COLOR, alpha=0.7)
axes[0,1].set_title('Tangency Portfolio Performance Metrics', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Value')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# 3. Tangency Portfolio vs Individual Stocks - Sharpe Ratios
individual_sharpe = [sharpe_ratio[stock] for stock in top_3_stocks]
all_sharpe = individual_sharpe + [tangency_sharpe]
all_labels = top_3_stocks + ['Tangency Portfolio']
colors = [LIGHT_COLOR] * len(top_3_stocks) + [PRIMARY_COLOR]
axes[1,0].bar(all_labels, all_sharpe, color=colors, alpha=0.7)
axes[1,0].set_title('Tangency Portfolio vs Individual Stocks (Sharpe Ratio)', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Sharpe Ratio')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# 4. Tangency Portfolio Risk-Return Position
axes[1,1].scatter(tangency_vol, tangency_return, s=300, alpha=0.8, color=SINGLE_COLOR, label='Tangency Portfolio')
# Add individual stocks for comparison
for i, stock in enumerate(top_3_stocks):
    axes[1,1].scatter(annualized_volatility[stock], annualized_return[stock], s=100, alpha=0.6, color=SINGLE_COLOR)
    axes[1,1].annotate(stock, (annualized_volatility[stock], annualized_return[stock]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1,1].set_title('Tangency Portfolio: Risk vs Return Position', fontweight='bold')
axes[1,1].set_xlabel('Volatility')
axes[1,1].set_ylabel('Expected Return')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step6a_tangency_portfolio.png', dpi=300, bbox_inches='tight')
plt.show()

# Tạo chart cho GMV và Equal Weight Portfolios
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. GMV and Equal Weight Portfolio Weights
gmv_ew_weights = pd.DataFrame({
    'GMV Portfolio': gmv_weights,
    'Equal Weight (Top3)': ew3_weights
}, index=top_3_stocks)
gmv_ew_weights.plot(kind='bar', ax=axes[0,0], width=0.8)
axes[0,0].set_title('GMV & Equal Weight Portfolio Weights (Top 3 Stocks)', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Weight')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,0].grid(True, alpha=0.3)

# 2. GMV and Equal Weight Performance - Sharpe Ratios
gmv_ew_performance = pd.DataFrame({
    'Expected Return': [gmv_return, ew3_return],
    'Volatility': [gmv_vol, ew3_vol],
    'Sharpe Ratio': [gmv_sharpe, ew3_sharpe]
}, index=['GMV Portfolio', 'Equal Weight (Top3)'])
axes[0,1].bar(gmv_ew_performance.index, gmv_ew_performance['Sharpe Ratio'], color=SINGLE_COLOR, alpha=0.7)
axes[0,1].set_title('GMV & Equal Weight Portfolio Sharpe Ratios', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Sharpe Ratio')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# 3. GMV and Equal Weight Performance - Returns
axes[1,0].bar(gmv_ew_performance.index, gmv_ew_performance['Expected Return'], color=SINGLE_COLOR, alpha=0.7)
axes[1,0].set_title('GMV & Equal Weight Portfolio Expected Returns', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Expected Return')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# 4. Risk-Return Scatter for GMV and Equal Weight Portfolios
axes[1,1].scatter(gmv_ew_performance['Volatility'], gmv_ew_performance['Expected Return'], s=200, alpha=0.7, color=SINGLE_COLOR)
for i, portfolio in enumerate(gmv_ew_performance.index):
    axes[1,1].annotate(portfolio, (gmv_ew_performance['Volatility'][portfolio], gmv_ew_performance['Expected Return'][portfolio]), 
                       xytext=(5, 5), textcoords='offset points')
axes[1,1].set_title('GMV & Equal Weight Portfolio: Risk vs Return', fontweight='bold')
axes[1,1].set_xlabel('Volatility')
axes[1,1].set_ylabel('Expected Return')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step6b_gmv_equal_weight_portfolios.png', dpi=300, bbox_inches='tight')
plt.show()

"""
FINAL SUMMARY
"""

print(f"\n" + "="*60)
print(f"📋 FINAL SUMMARY")
print(f"="*60)

print(f"\n🎯 QUANTITATIVE FLOW ANALYSIS COMPLETED!")
print(f"📁 Files created:")
print(f"   - step1_returns_analysis.png")
print(f"   - step2_volatility_analysis.png")
print(f"   - step3_risk_adjusted_metrics.png")
print(f"   - step4_ew_vs_cw_comparison.png")
print(f"   - step5_portfolio_weights_analysis.png")
print(f"   - step6_portfolio_optimization.png")
print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n🏆 KEY FINDINGS:")
print(f"   - Best Individual Stock: {sharpe_ratio.idxmax()} (Sharpe: {sharpe_ratio.max():.4f})")
print(f"   - Best Portfolio Strategy: {portfolio_performance['Sharpe Ratio'].idxmax()}")
print(f"   - Top 3 Stocks: {', '.join(top_3_stocks)}")
print(f"   - Tangency Portfolio Sharpe: {tangency_sharpe:.4f}")
print(f"   - Equal-Weighted vs Cap-Weighted: {'EW' if portfolio_comparison.loc['Sharpe', 'Equal-Weighted'] > portfolio_comparison.loc['Sharpe', 'Cap-Weighted'] else 'CW'} performs better")

"""
MACHINE LEARNING MODEL TRAINING
"""

print(f"\n🤖 MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

# Feature engineering function
def create_ml_features(price_data, df_return):
    """Create features for Random Forest"""
    features_list = []
    targets_list = []
    
    # Financial ratios (proxy values for energy sector)
    financial_ratios = pd.DataFrame({
        'ROA': [0.05, 0.08, 0.03, 0.04, 0.06, 0.07],
        'Leverage': [0.4, 0.3, 0.5, 0.35, 0.45, 0.25],
        'Cash_Ratio': [0.15, 0.20, 0.10, 0.18, 0.12, 0.22],
        'Asset_Turnover': [0.8, 1.0, 0.6, 0.9, 0.7, 1.1],
        'Current_Ratio': [1.2, 1.5, 1.0, 1.3, 1.1, 1.4],
        'Quick_Ratio': [0.9, 1.2, 0.7, 1.0, 0.8, 1.1]
    }, index=successful_tickers)
    
    for ticker in successful_tickers:
        print(f"🔧 Creating features for {ticker}...")
        
        prices = price_data[ticker].dropna()
        returns = df_return[ticker].dropna()
        
        # Align data
        common_dates = prices.index.intersection(returns.index)
        if len(common_dates) < 50:
            continue
            
        prices = prices.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Create features DataFrame
        features_df = pd.DataFrame(index=common_dates)
        
        # Technical indicators
        features_df['MA_5'] = prices.rolling(5).mean()
        features_df['MA_20'] = prices.rolling(20).mean()
        features_df['Price_to_MA5'] = prices / features_df['MA_5']
        
        # Return lags
        for lag in [1, 2, 3]:
            features_df[f'Return_Lag_{lag}'] = returns.shift(lag)
        
        # Volatility
        features_df['Volatility_10'] = returns.rolling(10).std()
        
        # Financial ratios
        features_df['ROA'] = financial_ratios['ROA'][ticker]
        features_df['Leverage'] = financial_ratios['Leverage'][ticker]
        features_df['Cash_Ratio'] = financial_ratios['Cash_Ratio'][ticker]
        features_df['Asset_Turnover'] = financial_ratios['Asset_Turnover'][ticker]
        features_df['Current_Ratio'] = financial_ratios['Current_Ratio'][ticker]
        features_df['Quick_Ratio'] = financial_ratios['Quick_Ratio'][ticker]
        features_df['Debt_Ratio'] = financial_ratios['Leverage'][ticker]
        
        # Target: next day return
        target = returns.shift(-1)
        
        # Remove NaN
        features_df = features_df.dropna()
        target = target.reindex(features_df.index).dropna()
        
        # Final alignment
        common_idx = features_df.index.intersection(target.index)
        if len(common_idx) < 30:
            continue
            
        features_final = features_df.loc[common_idx]
        target_final = target.loc[common_idx]
        
        # Add ticker identifier
        features_final['Ticker_' + ticker] = 1
        
        features_list.append(features_final)
        targets_list.append(target_final)
    
    if len(features_list) == 0:
        return None, None
        
    # Combine all features
    all_features = pd.concat(features_list, axis=0, sort=False).fillna(0)
    all_targets = pd.concat(targets_list, axis=0)
    
    return all_features, all_targets

# Train Random Forest model
print(f"\n🌲 Training Random Forest for daily return prediction...")
X, y = create_ml_features(price_data, df_return)

if X is None:
    print("❌ Không thể tạo features cho Random Forest")
    exit()

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X, y)
print(f"✅ Random Forest model trained successfully!")

"""
4-YEAR PREDICTION (2026-2030)
"""

print(f"\n🔮 4-YEAR PREDICTION (2026-2030)")
print("=" * 60)

# Tạo future dates cho 4 năm
future_years = [2026, 2027, 2028, 2029, 2030]
future_predictions = {}

# Get latest prices
latest_prices = {}
latest_returns = {}

for ticker in successful_tickers:
    latest_prices[ticker] = price_data[ticker].iloc[-1]
    latest_returns[ticker] = df_return[ticker].iloc[-1]

print(f"\n📊 Latest data (as of {price_data.index[-1].strftime('%Y-%m-%d')}):")
for ticker in successful_tickers:
    print(f"   {ticker}: Price={latest_prices[ticker]:.2f}")

# Predict for each year
for year in future_years:
    print(f"\n🔮 Predicting {year}...")
    
    # Tạo dates cho năm này
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    year_dates = pd.date_range(start=year_start, end=year_end, freq='D')
    year_business_days = year_dates[year_dates.weekday < 5]
    
    print(f"   📅 {len(year_business_days)} business days in {year}")
    
    year_predictions = {}
    
    for ticker in successful_tickers:
        # Simulate price evolution for this year
        current_price = latest_prices[ticker]
        np.random.seed(42 + year)  # Different seed for each year
        
        # Estimate drift and volatility from historical data
        historical_returns = df_return[ticker].dropna()
        drift = historical_returns.mean()
        volatility = historical_returns.std()
        
        # Generate future returns using geometric Brownian motion
        n_days = len(year_business_days)
        dt = 1/250
        random_shocks = np.random.normal(0, 1, n_days)
        future_returns = drift * dt + volatility * np.sqrt(dt) * random_shocks
        
        # Calculate future prices
        future_prices = [current_price]
        for ret in future_returns:
            future_prices.append(future_prices[-1] * (1 + ret))
        future_prices = np.array(future_prices[1:])
        
        # Create features for each day
        year_features_list = []
        for i, date in enumerate(year_business_days):
            if i >= len(future_prices):
                break
                
            features = {}
            
            # Technical indicators
            if i >= 4:
                ma_5 = np.mean(future_prices[i-4:i+1])
                features['MA_5'] = ma_5
                features['Price_to_MA5'] = future_prices[i] / ma_5
            else:
                features['MA_5'] = future_prices[i]
                features['Price_to_MA5'] = 1.0
                
            if i >= 19:
                ma_20 = np.mean(future_prices[i-19:i+1])
                features['MA_20'] = ma_20
            else:
                features['MA_20'] = future_prices[i]
            
            # Return lags
            if i >= 1:
                features['Return_Lag_1'] = future_returns[i-1]
            else:
                features['Return_Lag_1'] = latest_returns[ticker]
                
            if i >= 2:
                features['Return_Lag_2'] = future_returns[i-2]
            else:
                features['Return_Lag_2'] = latest_returns[ticker]
                
            if i >= 3:
                features['Return_Lag_3'] = future_returns[i-3]
            else:
                features['Return_Lag_3'] = latest_returns[ticker]
            
            # Volatility
            if i >= 9:
                features['Volatility_10'] = np.std(future_returns[i-9:i+1])
            else:
                features['Volatility_10'] = volatility
            
            # Financial ratios (constant)
            features['ROA'] = 0.05
            features['Leverage'] = 0.35
            features['Cash_Ratio'] = 0.15
            features['Asset_Turnover'] = 0.8
            features['Current_Ratio'] = 1.2
            features['Quick_Ratio'] = 0.9
            features['Debt_Ratio'] = 0.35
            
            # Ticker identifier
            features['Ticker_' + ticker] = 1
            for other_ticker in successful_tickers:
                if other_ticker != ticker:
                    features['Ticker_' + other_ticker] = 0
            
            year_features_list.append(features)
        
        # Convert to DataFrame and make predictions
        year_features_df = pd.DataFrame(year_features_list)
        
        # Ensure all columns from training are present
        for col in X.columns:
            if col not in year_features_df.columns:
                year_features_df[col] = 0
        
        year_features_df = year_features_df[X.columns]
        year_predictions[ticker] = rf_model.predict(year_features_df)
        
        # Calculate year statistics
        mean_return = np.mean(year_predictions[ticker])
        std_return = np.std(year_predictions[ticker])
        annual_return = mean_return * 250
        annual_vol = std_return * np.sqrt(250)
        sharpe = (annual_return - riskfree_rate) / annual_vol if annual_vol > 0 else 0
        
        print(f"   ✅ {ticker}: Return={annual_return:.2%}, Vol={annual_vol:.2%}, Sharpe={sharpe:.4f}")
    
    future_predictions[year] = year_predictions

"""
CREATE CHARTS FOR EACH YEAR
"""

print(f"\n🎨 Creating 6 ML Prediction Charts:")
print(f"   - 5 individual year charts (2026-2030)")
print(f"   - 1 comprehensive 5-year summary chart")

for year in future_years:
    print(f"\n📊 Creating chart for {year}...")
    
    year_predictions = future_predictions[year]
    
    # Calculate statistics for this year
    year_stats = {}
    for ticker in successful_tickers:
        predictions = year_predictions[ticker]
        mean_return = np.mean(predictions)
        std_return = np.std(predictions)
        annual_return = mean_return * 250
        annual_vol = std_return * np.sqrt(250)
        sharpe = (annual_return - riskfree_rate) / annual_vol if annual_vol > 0 else 0
        
        year_stats[ticker] = {
            'Annual_Return': annual_return,
            'Annual_Volatility': annual_vol,
            'Sharpe_Ratio': sharpe,
            'Mean_Daily_Return': mean_return,
            'Std_Daily_Return': std_return
        }
    
    # Create chart for this year
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Annual Returns
    returns = [year_stats[ticker]['Annual_Return'] for ticker in successful_tickers]
    colors = [MULTI_COLORS[0] if x > 0 else MULTI_COLORS[1] for x in returns]
    axes[0,0].bar(successful_tickers, returns, color=colors, alpha=0.7)
    axes[0,0].set_title(f'{year}: Annual Returns', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Annual Return')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Annual Volatility
    volatilities = [year_stats[ticker]['Annual_Volatility'] for ticker in successful_tickers]
    axes[0,1].bar(successful_tickers, volatilities, color=SINGLE_COLOR, alpha=0.7)
    axes[0,1].set_title(f'{year}: Annual Volatility', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Annual Volatility')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Sharpe Ratios
    sharpes = [year_stats[ticker]['Sharpe_Ratio'] for ticker in successful_tickers]
    colors = [ACCENT_COLOR if x > 0 else LIGHT_COLOR for x in sharpes]
    axes[0,2].bar(successful_tickers, sharpes, color=colors, alpha=0.7)
    axes[0,2].set_title(f'{year}: Sharpe Ratios', fontsize=14, fontweight='bold')
    axes[0,2].set_ylabel('Sharpe Ratio')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Risk vs Return Scatter
    axes[1,0].scatter(volatilities, returns, s=100, alpha=0.7, color=SINGLE_COLOR)
    for i, ticker in enumerate(successful_tickers):
        axes[1,0].annotate(ticker, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1,0].set_title(f'{year}: Risk vs Return', fontweight='bold')
    axes[1,0].set_xlabel('Volatility')
    axes[1,0].set_ylabel('Expected Return')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Top 3 Performers
    top_3_year = sorted(year_stats.items(), key=lambda x: x[1]['Sharpe_Ratio'], reverse=True)[:3]
    top_3_tickers = [item[0] for item in top_3_year]
    top_3_sharpes = [item[1]['Sharpe_Ratio'] for item in top_3_year]
    
    axes[1,1].bar(top_3_tickers, top_3_sharpes, color=SINGLE_COLOR, alpha=0.7)
    axes[1,1].set_title(f'{year}: Top 3 Performers', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    axes[1,2].axis('off')
    summary_text = f"{year} Performance Summary\n\n"
    for ticker in successful_tickers:
        stats = year_stats[ticker]
        summary_text += f"{ticker}: {stats['Annual_Return']:.2%} return, "
        summary_text += f"{stats['Annual_Volatility']:.2%} vol, "
        summary_text += f"{stats['Sharpe_Ratio']:.3f} Sharpe\n"
    
    axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1,2].set_title(f'{year}: Performance Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'year_{year}_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Chart saved: year_{year}_prediction.png")

"""
FINAL SUMMARY WITH ML PREDICTIONS
"""

print(f"\n📋 FINAL SUMMARY WITH ML PREDICTIONS")
print("=" * 60)
print(f"📊 TABLE: YEARLY PERFORMANCE SUMMARY (2026-2030)")

# Create summary table
summary_data = []
for year in future_years:
    year_stats = {}
    for ticker in successful_tickers:
        predictions = future_predictions[year][ticker]
        mean_return = np.mean(predictions)
        std_return = np.std(predictions)
        annual_return = mean_return * 250
        annual_vol = std_return * np.sqrt(250)
        sharpe = (annual_return - riskfree_rate) / annual_vol if annual_vol > 0 else 0
        
        year_stats[ticker] = {
            'Annual_Return': annual_return,
            'Annual_Volatility': annual_vol,
            'Sharpe_Ratio': sharpe
        }
    
    # Find best performer for this year
    best_ticker = max(year_stats.items(), key=lambda x: x[1]['Sharpe_Ratio'])[0]
    best_sharpe = year_stats[best_ticker]['Sharpe_Ratio']
    best_return = year_stats[best_ticker]['Annual_Return']
    
    summary_data.append({
        'Year': year,
        'Best_Performer': best_ticker,
        'Best_Sharpe': best_sharpe,
        'Best_Return': best_return,
        'Avg_Sharpe': np.mean([year_stats[ticker]['Sharpe_Ratio'] for ticker in successful_tickers]),
        'Avg_Return': np.mean([year_stats[ticker]['Annual_Return'] for ticker in successful_tickers])
    })

summary_df = pd.DataFrame(summary_data)

print("📊 TABLE: YEARLY PERFORMANCE SUMMARY")
print("-" * 80)
print(summary_df.round(4))

"""
ML RESULTS TABLES - 6 STOCKS SEPARATE TABLES
"""

print(f"\n📊 ML RESULTS TABLES - 6 STOCKS SEPARATE")
print("=" * 60)
print(f"📊 TABLE 1: ML DAILY RETURNS (Annualized) - 6 STOCKS (2026-2030)")

# Create separate tables for ML results
ml_daily_returns = {}
ml_sharpe_ratios = {}

for ticker in successful_tickers:
    daily_returns_list = []
    sharpe_ratios_list = []
    
    for year in future_years:
        predictions = future_predictions[year][ticker]
        mean_return = np.mean(predictions)
        std_return = np.std(predictions)
        annual_return = mean_return * 250
        annual_vol = std_return * np.sqrt(250)
        sharpe = (annual_return - riskfree_rate) / annual_vol if annual_vol > 0 else 0
        
        daily_returns_list.append(annual_return)
        sharpe_ratios_list.append(sharpe)
    
    ml_daily_returns[ticker] = daily_returns_list
    ml_sharpe_ratios[ticker] = sharpe_ratios_list

# Create DataFrames
ml_daily_returns_df = pd.DataFrame(ml_daily_returns, index=future_years)
ml_sharpe_ratios_df = pd.DataFrame(ml_sharpe_ratios, index=future_years)

print("📊 TABLE 1: ML DAILY RETURNS (Annualized) - 6 STOCKS (2026-2030)")
print("-" * 80)
print(ml_daily_returns_df.round(4))

print("\n📊 TABLE 2: ML SHARPE RATIOS - 6 STOCKS (2026-2030)")
print("-" * 80)
print(ml_sharpe_ratios_df.round(4))

"""
CREATE 2 SEPARATE CHARTS FOR ML RESULTS
"""

print(f"\n🎨 Creating 2 separate charts for ML results...")

# Chart 1: ML Daily Returns (Annualized) - 6 Stocks
fig1, ax1 = plt.subplots(figsize=(15, 10))

# Plot each stock's returns over 5 years
for i, ticker in enumerate(successful_tickers):
    ax1.plot(future_years, ml_daily_returns_df[ticker], marker='o', linewidth=2, 
             label=f'{ticker} ({company_names[ticker]})', markersize=6, color=MULTI_COLORS[i % len(MULTI_COLORS)])

ax1.set_title('ML Daily Returns (Annualized) - 6 Stocks (2026-2030)', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Annualized Return', fontsize=12)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Add value annotations
for ticker in successful_tickers:
    for i, year in enumerate(future_years):
        value = ml_daily_returns_df[ticker].iloc[i]
        ax1.annotate(f'{value:.1%}', (year, value), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('ml_daily_returns_6stocks_4years.png', dpi=300, bbox_inches='tight')
plt.show()

print("   ✅ Chart 1 saved: ml_daily_returns_6stocks_4years.png")

# Chart 2: ML Sharpe Ratios - 6 Stocks
fig2, ax2 = plt.subplots(figsize=(15, 10))

# Plot each stock's Sharpe ratios over 5 years
for i, ticker in enumerate(successful_tickers):
    ax2.plot(future_years, ml_sharpe_ratios_df[ticker], marker='s', linewidth=2, 
             label=f'{ticker} ({company_names[ticker]})', markersize=6, color=MULTI_COLORS[i % len(MULTI_COLORS)])

ax2.set_title('ML Sharpe Ratios - 6 Stocks (2026-2030)', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Sharpe Ratio', fontsize=12)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Add value annotations
for ticker in successful_tickers:
    for i, year in enumerate(future_years):
        value = ml_sharpe_ratios_df[ticker].iloc[i]
        ax2.annotate(f'{value:.2f}', (year, value), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('ml_sharpe_ratios_6stocks_4years.png', dpi=300, bbox_inches='tight')
plt.show()

print("   ✅ Chart 2 saved: ml_sharpe_ratios_6stocks_4years.png")

"""
COMPARISON CHARTS: QUANTITATIVE vs ML MODELS
"""

print(f"\n📊 Creating comparison charts: Quantitative vs ML Models...")

# Prepare historical quantitative data for comparison
historical_annual_returns = annualized_return
historical_sharpe_ratios = sharpe_ratio

# Calculate average ML predictions across 5 years
ml_avg_returns = ml_daily_returns_df.mean()
ml_avg_sharpes = ml_sharpe_ratios_df.mean()

# Chart 1: Daily Returns Comparison (Quantitative vs ML)
fig1, ax1 = plt.subplots(figsize=(15, 10))

# Prepare data for comparison
tickers_list = list(successful_tickers)
quanti_returns = [historical_annual_returns[ticker] for ticker in tickers_list]
ml_returns = [ml_avg_returns[ticker] for ticker in tickers_list]

x_pos = np.arange(len(tickers_list))
width = 0.35

# Create bars
bars1 = ax1.bar(x_pos - width/2, quanti_returns, width, label='Quantitative Model', 
                alpha=0.7, color=MULTI_COLORS[0])
bars2 = ax1.bar(x_pos + width/2, ml_returns, width, label='ML Model (Avg 2025-2030)',
                alpha=0.7, color=MULTI_COLORS[1])

ax1.set_title('Daily Returns Comparison: Quantitative vs ML Models', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Stocks', fontsize=12)
ax1.set_ylabel('Annualized Return', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(tickers_list)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + (0.01 if height1 >= 0 else -0.02),
             f'{quanti_returns[i]:.1%}', ha='center', va='bottom' if height1 >= 0 else 'top', fontsize=9)
    ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + (0.01 if height2 >= 0 else -0.02),
             f'{ml_returns[i]:.1%}', ha='center', va='bottom' if height2 >= 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('comparison_quanti_vs_ml_daily_returns.png', dpi=300, bbox_inches='tight')
plt.show()

print("   ✅ Comparison Chart 1 saved: comparison_quanti_vs_ml_daily_returns.png")

# Chart 2: Sharpe Ratios Comparison (Quantitative vs ML)
fig2, ax2 = plt.subplots(figsize=(15, 10))

# Prepare data for comparison
quanti_sharpes = [historical_sharpe_ratios[ticker] for ticker in tickers_list]
ml_sharpes = [ml_avg_sharpes[ticker] for ticker in tickers_list]

# Create bars
bars1 = ax2.bar(x_pos - width/2, quanti_sharpes, width, label='Quantitative Model', 
                alpha=0.7, color=MULTI_COLORS[0])
bars2 = ax2.bar(x_pos + width/2, ml_sharpes, width, label='ML Model (Avg 2025-2030)', 
                alpha=0.7, color=MULTI_COLORS[1])

ax2.set_title('Sharpe Ratios Comparison: Quantitative vs ML Models', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Stocks', fontsize=12)
ax2.set_ylabel('Sharpe Ratio', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(tickers_list)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax2.text(bar1.get_x() + bar1.get_width()/2., height1 + (0.01 if height1 >= 0 else -0.01),
             f'{quanti_sharpes[i]:.3f}', ha='center', va='bottom' if height1 >= 0 else 'top', fontsize=9)
    ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + (0.01 if height2 >= 0 else -0.01),
             f'{ml_sharpes[i]:.1f}', ha='center', va='bottom' if height2 >= 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('comparison_quanti_vs_ml_sharpe_ratios.png', dpi=300, bbox_inches='tight')
plt.show()

print("   ✅ Comparison Chart 2 saved: comparison_quanti_vs_ml_sharpe_ratios.png")

# Print comparison summary
print(f"\n📊 COMPARISON SUMMARY: QUANTITATIVE vs ML MODELS")
print("=" * 60)

comparison_summary = pd.DataFrame({
    'Stock': tickers_list,
    'Quantitative_Return': quanti_returns,
    'ML_Return': ml_returns,
    'Return_Difference': [ml - quanti for quanti, ml in zip(quanti_returns, ml_returns)],
    'Quantitative_Sharpe': quanti_sharpes,
    'ML_Sharpe': ml_sharpes,
    'Sharpe_Difference': [ml - quanti for quanti, ml in zip(quanti_sharpes, ml_sharpes)]
})

print("📊 TABLE: MODEL COMPARISON SUMMARY")
print("-" * 100)
print(comparison_summary.round(4))

# Find best performers in each model
print(f"\n🏆 BEST PERFORMERS:")
print(f"   📊 Quantitative Model:")
print(f"      - Best Return: {comparison_summary.loc[comparison_summary['Quantitative_Return'].idxmax(), 'Stock']} ({comparison_summary['Quantitative_Return'].max():.2%})")
print(f"      - Best Sharpe: {comparison_summary.loc[comparison_summary['Quantitative_Sharpe'].idxmax(), 'Stock']} ({comparison_summary['Quantitative_Sharpe'].max():.4f})")
print(f"   🤖 ML Model:")
print(f"      - Best Return: {comparison_summary.loc[comparison_summary['ML_Return'].idxmax(), 'Stock']} ({comparison_summary['ML_Return'].max():.2%})")
print(f"      - Best Sharpe: {comparison_summary.loc[comparison_summary['ML_Sharpe'].idxmax(), 'Stock']} ({comparison_summary['ML_Sharpe'].max():.1f})")

print(f"\n📈 MODEL AGREEMENT:")
# Count how many stocks have same best performer
quanti_best_return = comparison_summary.loc[comparison_summary['Quantitative_Return'].idxmax(), 'Stock']
ml_best_return = comparison_summary.loc[comparison_summary['ML_Return'].idxmax(), 'Stock']
quanti_best_sharpe = comparison_summary.loc[comparison_summary['Quantitative_Sharpe'].idxmax(), 'Stock']
ml_best_sharpe = comparison_summary.loc[comparison_summary['ML_Sharpe'].idxmax(), 'Stock']

print(f"   - Best Return Agreement: {'✅ YES' if quanti_best_return == ml_best_return else '❌ NO'} ({quanti_best_return} vs {ml_best_return})")
print(f"   - Best Sharpe Agreement: {'✅ YES' if quanti_best_sharpe == ml_best_sharpe else '❌ NO'} ({quanti_best_sharpe} vs {ml_best_sharpe})")

# Create final summary chart
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Best Performer by Year
axes[0,0].bar(summary_df['Year'], summary_df['Best_Sharpe'], color=SINGLE_COLOR, alpha=0.7)
axes[0,0].set_title('Best Sharpe Ratio by Year', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Best Sharpe Ratio')
axes[0,0].set_xlabel('Year')
axes[0,0].grid(True, alpha=0.3)

# 2. Average Performance by Year
axes[0,1].plot(summary_df['Year'], summary_df['Avg_Sharpe'], marker='o', linewidth=2, label='Avg Sharpe', color=MULTI_COLORS[0])
axes[0,1].plot(summary_df['Year'], summary_df['Avg_Return'], marker='s', linewidth=2, label='Avg Return', color=MULTI_COLORS[1])
axes[0,1].set_title('Average Performance by Year (2026-2030)', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Value')
axes[0,1].set_xlabel('Year')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Best Performers
best_performers = summary_df['Best_Performer'].value_counts()
axes[1,0].bar(best_performers.index, best_performers.values, color=SINGLE_COLOR, alpha=0.7)
axes[1,0].set_title('Best Performer Frequency (2026-2030)', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Number of Years')
axes[1,0].set_xlabel('Stock')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# 4. Performance Trend
axes[1,1].plot(summary_df['Year'], summary_df['Best_Return'], marker='o', linewidth=2, label='Best Return', color=MULTI_COLORS[0])
axes[1,1].plot(summary_df['Year'], summary_df['Avg_Return'], marker='s', linewidth=2, label='Avg Return', color=MULTI_COLORS[1])
axes[1,1].set_title('Return Trends (2026-2030)', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Return')
axes[1,1].set_xlabel('Year')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('4year_summary_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Creating comprehensive 5-year summary chart...")
print(f"   ✅ Chart 6 saved: 4year_summary_prediction.png")

print(f"\n🎉 COMPLETE QUANTITATIVE + ML ANALYSIS COMPLETED!")
print(f"📁 Files created:")
print(f"   📊 Quantitative Flow Charts:")
print(f"   - step1_returns_analysis.png")
print(f"   - step2_volatility_analysis.png")
print(f"   - step3_risk_adjusted_metrics.png")
print(f"   - step4a_equal_weighted_portfolio.png")
print(f"   - step4b_capitalization_weighted_portfolio.png")
print(f"   - step5_portfolio_weights_analysis.png")
print(f"   - step6a_tangency_portfolio.png")
print(f"   - step6b_gmv_equal_weight_portfolios.png")
print(f"   🔮 ML Prediction Charts (6 total):")
for year in future_years:
    print(f"   - year_{year}_prediction.png")
print(f"   - 4year_summary_prediction.png (5-year comprehensive summary)")
print(f"   📈 ML Results Charts:")
print(f"   - ml_daily_returns_6stocks_4years.png")
print(f"   - ml_sharpe_ratios_6stocks_4years.png")
print(f"   🔄 Model Comparison Charts:")
print(f"   - comparison_quanti_vs_ml_daily_returns.png")
print(f"   - comparison_quanti_vs_ml_sharpe_ratios.png")
print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n🏆 KEY FINDINGS:")
print(f"   📊 Historical Analysis:")
print(f"   - Best Individual Stock: {sharpe_ratio.idxmax()} (Sharpe: {sharpe_ratio.max():.4f})")
print(f"   - Best Portfolio Strategy: {portfolio_performance['Sharpe Ratio'].idxmax()}")
print(f"   - Top 3 Stocks: {', '.join(top_3_stocks)}")
print(f"   - Tangency Portfolio Sharpe: {tangency_sharpe:.4f}")
print(f"   🔮 Future Predictions (2026-2030):")
print(f"   - Most frequent best performer: {best_performers.idxmax()} ({best_performers.max()} years)")
print(f"   - Best average Sharpe: {summary_df['Avg_Sharpe'].max():.4f} in {summary_df.loc[summary_df['Avg_Sharpe'].idxmax(), 'Year']}")
print(f"   - Highest return: {summary_df['Best_Return'].max():.2%} in {summary_df.loc[summary_df['Best_Return'].idxmax(), 'Year']}")
print(f"   - Model used: Random Forest with 20+ features")
print(f"   - Prediction period: 2026-2030 (4 years)")
