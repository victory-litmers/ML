# Vietnamese Energy Stocks Prediction Model

## 📊 Giới thiệu

Dự án này phát triển một mô hình Machine Learning để dự đoán hiệu suất của các cổ phiếu năng lượng Việt Nam, bao gồm phân tích định lượng toàn diện và đánh giá độ tin cậy của mô hình.

### 🎯 Mục tiêu

- Phân tích hiệu suất 6 cổ phiếu năng lượng Việt Nam (PLX, OIL, GAS, PPC, GEG, POW)
- Xây dựng mô hình ML dự đoán với độ tin cậy cao
- Tối ưu hóa danh mục đầu tư dựa trên phân tích rủi ro-lợi nhuận
- Cung cấp dự đoán 5 năm (2026-2030)

## 🏗️ Cấu trúc dự án

```
ML/
├── complete_real_data_model.py          # Mô hình chính với tất cả tính năng
├── FINAL_RELIABILITY_ASSESSMENT.md      # Báo cáo đánh giá độ tin cậy
├── FINAL_SUMMARY.md                     # Tổng kết dự án
├── requirements.txt                     # Dependencies
├── vnindex.xlsx                        # Dữ liệu VN-Index
├── README.md                           # File này
└── charts/                             # Các biểu đồ kết quả
    ├── *_detailed_analysis.png         # Phân tích chi tiết từng cổ phiếu
    ├── *_chart.png                     # Các biểu đồ phân tích
    └── ml_prediction_*.png             # Dự đoán ML
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- 8GB RAM (khuyến nghị)
- Kết nối internet để tải dữ liệu

### Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd ML

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt thêm vnstock (tùy chọn)
pip install -U vnstock
```

### Dependencies chính

- `pandas` - Xử lý dữ liệu
- `numpy` - Tính toán số học
- `scikit-learn` - Machine Learning
- `matplotlib` - Vẽ biểu đồ
- `seaborn` - Visualization
- `yfinance` - Dữ liệu tài chính
- `vnstock` - Dữ liệu Việt Nam (tùy chọn)

## 📖 Hướng dẫn sử dụng

### Chạy mô hình hoàn chỉnh

```bash
python complete_real_data_model.py
```

### Kết quả mong đợi

Chương trình sẽ thực hiện:

1. **Thu thập dữ liệu**:

   - Dữ liệu giá cổ phiếu từ vnstock
   - Dữ liệu tài chính quarterly
   - Dữ liệu VN-Index
   - Chỉ số kinh tế vĩ mô (nếu có)

2. **Phân tích định lượng**:

   - Return & Risk Analysis
   - Sharpe Ratio & Risk-Adjusted Metrics
   - Portfolio Optimization
   - Efficient Frontier Analysis

3. **Machine Learning**:

   - Feature Engineering (36 features)
   - Model Selection (10 models)
   - Ensemble Methods
   - Walk-forward Validation

4. **Dự đoán tương lai**:
   - Dự đoán 2026-2030
   - Statistical approach
   - Risk assessment

### Output files

- **Charts**: 20+ biểu đồ phân tích
- **Console output**: Kết quả chi tiết
- **Performance metrics**: R², RMSE, MAE, Sharpe ratios

## 📊 Tính năng chính

### 1. Phân tích định lượng

- **Return Analysis**: Daily, monthly, annual returns
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown
- **Sharpe Ratios**: Risk-adjusted performance
- **Portfolio Optimization**: Efficient frontier, tangency portfolio

### 2. Machine Learning

- **Feature Engineering**: 36 features bao gồm:

  - Technical indicators (RSI, Bollinger Bands, MA ratios)
  - Volatility measures (5, 20, 60-day)
  - Market interaction features
  - Economic indicators
  - Fundamental ratios

- **Model Selection**: 10 models với Time Series CV
- **Ensemble Methods**: Top 3 models với weighted prediction
- **Validation**: Walk-forward validation

### 3. Đánh giá độ tin cậy

- **Cross-validation**: K-Fold và Time Series CV
- **Feature Stability**: Bootstrap analysis
- **Model Calibration**: Calibration assessment
- **Prediction Uncertainty**: Confidence intervals

## 📈 Kết quả chính

### Hiệu suất mô hình

- **R² Score**: 0.2633 (cải thiện từ 0.0008)
- **RMSE**: 0.0185 (cải thiện từ 0.9999)
- **Reliability Grade**: B (High Reliability)
- **Walk-Forward CV R²**: 0.0668 ± 0.0392

### Top performers

- **Best Individual Stock**: OIL (Sharpe: 0.1475)
- **Best Portfolio**: RTRR Portfolio (Sharpe: 0.1562)
- **Equal-Weighted vs Cap-Weighted**: 0.0822 vs 0.0738

### Dự đoán 2026-2030

- **OIL**: 23.41% → 26.24% (2026-2030)
- **POW**: 13.33% → 10.36% (2026-2030)
- **GAS**: 7.26% → 12.14% (2026-2030)

## 🔧 Tùy chỉnh

### Thay đổi cổ phiếu

Chỉnh sửa `tickers` trong `complete_real_data_model.py`:

```python
tickers = ['PLX', 'OIL', 'GAS', 'PPC', 'GEG', 'POW']
```

### Thay đổi thời gian

Chỉnh sửa `start_date` và `end_date`:

```python
start_date = '2020-01-01'
end_date = '2025-08-15'
```

### Thay đổi models

Chỉnh sửa `models` dictionary trong function `train_improved_models`:

```python
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(),
    # Thêm models khác...
}
```

## 📋 Troubleshooting

### Lỗi thường gặp

1. **ImportError: No module named 'vnstock'**

   ```bash
   pip install -U vnstock
   ```

2. **Memory Error**

   - Giảm số lượng features
   - Giảm thời gian phân tích
   - Tăng RAM

3. **Data not found**
   - Kiểm tra kết nối internet
   - Kiểm tra ticker symbols
   - Kiểm tra date range

### Performance tips

- Sử dụng SSD để tăng tốc I/O
- Tăng RAM để xử lý dữ liệu lớn
- Sử dụng GPU cho deep learning (nếu có)

## 📊 Báo cáo độ tin cậy

Xem file `FINAL_RELIABILITY_ASSESSMENT.md` để biết chi tiết về:

- Đánh giá độ tin cậy mô hình
- Các cải thiện đã thực hiện
- Khuyến nghị triển khai
- So sánh trước/sau cải thiện

## 🤝 Đóng góp

### Cách đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

### Guidelines

- Code style: PEP 8
- Documentation: Docstrings cho functions
- Testing: Unit tests cho core functions
- Performance: Optimize cho large datasets

## 📞 Liên hệ

- **Author**: Vinh Nguyen
- **Email**: [your-email@domain.com]
- **GitHub**: [your-github-username]
- **LinkedIn**: [your-linkedin-profile]

## 📄 License

MIT License - Xem file LICENSE để biết chi tiết.

## 🙏 Acknowledgments

- **vnstock**: Dữ liệu tài chính Việt Nam
- **yfinance**: Dữ liệu tài chính global
- **scikit-learn**: Machine Learning framework
- **pandas**: Data manipulation
- **matplotlib/seaborn**: Visualization

---

**Lưu ý**: Mô hình này chỉ dành cho mục đích nghiên cứu và giáo dục. Không sử dụng cho mục đích đầu tư thực tế mà không có đánh giá rủi ro kỹ lưỡng.
