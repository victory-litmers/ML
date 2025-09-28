# Complete Portfolio Optimization with Real-Time Financial Data - ENHANCED VERSION

## Ứng dụng phương pháp định lượng hoàn chỉnh và mô hình học máy với dữ liệu tài chính quarterly thực

### 🎯 Tổng quan Dự án

Dự án này triển khai một khung tối ưu hóa danh mục toàn diện kết hợp **Quantitative Flow Analysis hoàn chỉnh** với **mô hình học máy Random Forest** và **dữ liệu tài chính quarterly thực** để phân tích và tối ưu hóa danh mục đầu tư cho các công ty năng lượng Việt Nam.

**🔥 ENHANCED VERSION**: Tích hợp hoàn toàn **dữ liệu quarterly timeseries thực** từ vnstock cho 6 chỉ số tài chính quan trọng, **Complete Quantitative Flow Analysis** theo chuẩn academic, và **17 professional charts** với dự đoán ML 5 năm (2026-2030).

### 🏢 Công ty Mục tiêu

Phân tích tập trung vào 6 công ty năng lượng hàng đầu Việt Nam:

| Ticker | Tên Công ty        | Lĩnh vực  | Quarterly Data Coverage         |
| ------ | ------------------ | --------- | ------------------------------- |
| PLX    | Petrolimex         | Dầu khí   | 22 quarters (2020-Q3 → 2025-Q2) |
| OIL    | PVOIL              | Dầu khí   | 22 quarters (2020-Q3 → 2025-Q2) |
| GAS    | PetroVietnam Gas   | Dầu khí   | 22 quarters (2020-Q3 → 2025-Q2) |
| PPC    | Nhiệt điện Phả Lại | Phát điện | 22 quarters (2020-Q3 → 2025-Q2) |
| GEG    | CTCP Điện Gia Lai  | Phát điện | 22 quarters (2020-Q3 → 2025-Q2) |
| POW    | PV Power           | Phát điện | 22 quarters (2020-Q3 → 2025-Q2) |

### 📊 Phương pháp Luận Hoàn Chỉnh

#### 1. **Dữ liệu Thực 100% (NO MOCK DATA)**

**📊 6 CHỈ SỐ TÀI CHÍNH QUARTERLY THỰC:**

1. **Khối lượng giao dịch (triệu)** - Từ VN-Index real-time
2. **Chỉ số thị trường (VN-Index)** - VNINDEX từ vnstock
3. **Tỷ lệ nợ (Leverage)** - D/E Ratio từ quarterly financial statements
4. **Lợi nhuận trên tổng tài sản (ROA)** - Net Income/Average Total Assets
5. **Tỷ lệ tiền mặt (Cash Ratio)** - Cash+Equivalents/Current Liabilities
6. **Asset Turnover** - Net Sales/Average Total Assets

**📈 Quarterly Timeseries Data:**

- **22 quarters mỗi stock** từ 2020-Q3 đến 2025-Q2
- **Perfect time alignment** với price data (2020-2025)
- **Dynamic quarterly trends** thay vì static values
- **QoQ change features** để capture xu hướng

#### 2. **Complete Quantitative Flow Analysis (4 Steps)**

**🔢 STEP 1: RETURN & RISK ANALYSIS**

- **Returns**: Daily, Cumulative, Annualized
- **Risk Metrics**: Volatility, Semideviation
- **Advanced Risk**: VaR (Historic & Gaussian), CVaR
- **Statistical**: Skewness, Kurtosis

**🔢 STEP 2: SHARPE RATIO & RISK-ADJUSTED METRICS**

- **Sharpe Ratio**: Complete ranking analysis
- **Drawdown Analysis**: Maximum drawdown cho mỗi stock
- **Wealth Index**: Cumulative performance tracking
- **Risk-adjusted**: Multiple metrics

**🔢 STEP 3: FACTOR ANALYSIS - EW vs CW PORTFOLIOS**

- **Equal-Weighted Portfolio**: 16.67% mỗi stock
- **Capitalization-Weighted Portfolio**: Theo market cap proxy
- **Performance Comparison**: Detailed analysis
- **Outperformance**: EW vs CW metrics

**🔢 STEP 4: PORTFOLIO OPTIMIZATION**

- **Efficient Frontier**: 25 points computation
- **Tangency Portfolio**: Maximum Sharpe optimization
- **GMV Portfolio**: Global Minimum Variance
- **Portfolio Weights**: Complete allocation tables

#### 3. **Enhanced Machine Learning Model**

**🤖 31 Features toàn diện:**

- **Technical Indicators (7)**: MA_5, MA_20, Price_to_MA5, Return_Lags, Volatility
- **VN-Index Features (4)**: Close, Return, Return_Lag1, Volatility
- **Volume Features (3)**: Volume_Millions, Volume_MA_5, Volume_Ratio
- **Quarterly Financial Ratios (7)**: ROA, Leverage, Cash_Ratio, Asset_Turnover, Current_Ratio, Quick_Ratio, Debt_Ratio
- **QoQ Change Features (4)**: ROA_QoQ_Change, Leverage_QoQ_Change, Cash_Ratio_QoQ_Change, Asset_Turnover_QoQ_Change
- **Ticker Features (6)**: Stock identification variables

**📈 Model Performance:**

- **R² Score**: 0.096289
- **RMSE**: 0.021211
- **Real Data Coverage**: 100% (6/6 stocks)
- **Features**: 8,122 samples × 31 features

#### 4. **Future Predictions với Quarterly Trends (2026-2030)**

**🔮 ML Predictions sử dụng:**

- **Quarterly trend analysis** từ historical data
- **Time series forecasting** cho financial ratios
- **Random Forest model** với quarterly features
- **5-year predictions** với yearly breakdown

### 🎨 Professional Visualization - 17 Charts

#### **📊 Core Quantitative Charts (7 charts):**

1. **`sharpe_ratio_chart.png`** - Sharpe Ratio ranking bar chart
2. **`wealth_index_chart.png`** - Cumulative wealth evolution
3. **`drawdown_analysis_chart.png`** - 6-subplot drawdown analysis
4. **`portfolio_weights_chart.png`** - EW vs GMV vs Tangency weights
5. **`efficient_frontier_chart.png`** - Efficient frontier với portfolio points
6. **`risk_return_scatter_chart.png`** - Risk-return scatter với Sharpe color coding

#### **⚖️ Factor Analysis Charts (4 charts):**

7. **`equal_weighted_performance_chart.png`** - EW portfolio riêng biệt
8. **`cap_weighted_performance_chart.png`** - CW portfolio riêng biệt
9. **`ew_vs_cw_comparison_chart.png`** - EW vs CW comparison
10. **`tangency_portfolio_analysis_chart.png`** - Tangency analysis với weights + efficient frontier

#### **🔮 ML Prediction Charts (6 charts):**

11. **`ml_prediction_2026_chart.png`** - Predictions cho 2026
12. **`ml_prediction_2027_chart.png`** - Predictions cho 2027
13. **`ml_prediction_2028_chart.png`** - Predictions cho 2028
14. **`ml_prediction_2029_chart.png`** - Predictions cho 2029
15. **`ml_prediction_2030_chart.png`** - Predictions cho 2030
16. **`ml_prediction_summary_5years_chart.png`** - 5-year heatmap summary

**Tất cả charts:**

- **300 DPI resolution** cho print quality
- **Professional styling** với consistent themes
- **Comprehensive annotations** và value labels
- **Color-coded insights** dựa trên performance metrics

---

**📅 Analysis Period**: 2020-2025 | **Updated**: September 2025 | **Framework**: Complete Quantitative + ML + Real Quarterly Data | **Version**: ENHANCED (17 Charts + 100% Real Data)\*\*
