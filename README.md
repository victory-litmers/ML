# Portfolio Optimization for Vietnamese Energy Firms - VERSION 4

## Ứng dụng phương pháp định lượng và mô hình học máy trong tối ưu hóa danh mục đầu tư

### 🎯 Tổng quan Dự án

Dự án này triển khai một khung tối ưu hóa danh mục toàn diện kết hợp **phương pháp định lượng** với **mô hình học máy** để phân tích và tối ưu hóa danh mục đầu tư cho các công ty năng lượng Việt Nam. Nghiên cứu so sánh các phương pháp định lượng truyền thống với dự đoán học máy Random Forest để cung cấp các chiến lược đầu tư vượt trội.

**🔥 VERSION 4 ENHANCEMENT**: Phân tích biến định lượng hoàn chỉnh với **10 danh mục toàn diện** và **14 chỉ số cốt lõi cho mỗi cổ phiếu**, cộng với **6 biến thị trường quan trọng** bao gồm tương quan VN-Index và phân tích khối lượng. **Dự đoán tương lai 4 năm (2026-2030)** với mô hình Random Forest.

### 🏢 Công ty Mục tiêu

Phân tích tập trung vào 6 công ty năng lượng hàng đầu Việt Nam:

| Ticker | Tên Công ty          | Lĩnh vực  | Tương quan VN-Index |
| ------ | -------------------- | --------- | ------------------- |
| PLX    | Petrolimex           | Dầu khí   | 0.6165              |
| OIL    | PVOIL                | Dầu khí   | 0.5895              |
| COM    | CTCP Vật tư Xăng Dầu | Dầu khí   | 0.1043              |
| PPC    | Nhiệt điện Phả Lại   | Phát điện | 0.5019              |
| GEG    | CTCP Điện Gia Lai    | Phát điện | 0.5185              |
| POW    | PV Power             | Phát điện | 0.6203              |

### 📊 Phương pháp Luận

#### 1. **Mô hình Định lượng (Traditional Portfolio Theory)**

- **Phân tích Lợi nhuận & Rủi ro**: Tính toán lợi nhuận hàng ngày, ước lượng độ biến động, các chỉ số điều chỉnh rủi ro
- **Phân tích Yếu tố**: So sánh danh mục Trọng số đều vs Trọng số vốn hóa
- **Tối ưu hóa Danh mục**: Tối đa hóa Tỷ lệ Sharpe, Phương sai tối thiểu toàn cục (GMV), Danh mục Trọng số đều
- **Tính toán Tỷ lệ Sharpe**: Sử dụng lãi suất phi rủi ro 2.7%
- **📊 BIẾN ĐỊNH LƯỢNG TOÀN DIỆN (10 Danh mục)**:
  - **Biến Lợi nhuận**: Lợi nhuận Hàng ngày, Hàng tháng, Hàng năm, Tích lũy, Vượt trội
  - **Biến Rủi ro**: Độ lệch chuẩn, Phương sai, Độ biến động, Độ lệch bán, Độ biến động hàng năm
  - **Hiệu suất Điều chỉnh Rủi ro**: Tỷ lệ Sharpe, Tỷ lệ Lợi nhuận-Rủi ro
  - **Biến Danh mục**: Trọng số và lợi nhuận Trọng số đều & Trọng số vốn hóa
  - **Biến Tương quan**: Ma trận tương quan 6x6
  - **Biến Hiệp phương sai**: Ma trận hiệp phương sai 6x6
  - **Chỉ số Bổ sung**: VaR (95%), CVaR (95%), Sụt giảm tối đa, Độ lệch, Độ nhọn
  - **Biến Thị trường & Khối lượng**: Thống kê khối lượng (triệu), Tương quan VN-Index
  - **Tỷ số Tài chính**: ROA, Đòn bẩy, Tỷ số Tiền mặt, Vòng quay Tài sản, Tỷ số Thanh khoản, Tỷ số Nhanh
  - **Bảng Tóm tắt**: 14 chỉ số cốt lõi cho mỗi cổ phiếu

#### 2. **Mô hình Random Forest (Machine Learning Approach)**

- **Kỹ thuật Đặc trưng**: **20 biến toàn diện** qua 4 danh mục
  - **Chỉ báo Kỹ thuật (4 biến)**:
    - `MA_5`: Trung bình động 5 ngày
    - `MA_20`: Trung bình động 20 ngày
    - `Price_to_MA5`: Tỷ lệ giá/MA5
    - `Volatility_10`: Độ biến động 10 ngày
  - **Độ trễ Lợi nhuận (3 biến)**:
    - `Return_Lag_1`: Lợi nhuận ngày trước
    - `Return_Lag_2`: Lợi nhuận 2 ngày trước
    - `Return_Lag_3`: Lợi nhuận 3 ngày trước
  - **Tỷ số Tài chính (7 biến)**:
    - `ROA`: Lợi nhuận trên Tài sản
    - `Leverage`: Tỷ lệ Nợ/Vốn chủ sở hữu
    - `Cash_Ratio`: Tỷ lệ Tiền mặt/Nợ ngắn hạn
    - `Asset_Turnover`: Vòng quay Tài sản
    - `Current_Ratio`: Tỷ lệ Thanh khoản
    - `Quick_Ratio`: Tỷ lệ Thanh khoản nhanh
    - `Debt_Ratio`: Tỷ lệ Nợ
  - **Định danh Cổ phiếu (6 biến)**:
    - `Ticker_PLX`, `Ticker_OIL`, `Ticker_COM`, `Ticker_PPC`, `Ticker_GEG`, `Ticker_POW`
- **Cấu hình Mô hình**:
  - **Thuật toán**: Random Forest Regressor
  - **Số cây**: 100
  - **Độ sâu tối đa**: 10
  - **Mục tiêu**: Lợi nhuận Ngày tiếp theo
  - **Đặc trưng**: 20 biến
  - **Mẫu**: 8,070 quan sát
- **Tầm quan trọng Đặc trưng (Top 5)**:
  1. `Price_to_MA5`: 28.31%
  2. `Return_Lag_1`: 13.02%
  3. `Return_Lag_2`: 12.50%
  4. `MA_20`: 11.76%
  5. `Return_Lag_3`: 11.76%
- **Nguồn Dữ liệu**:
  - **Chính**: vnstock (Dữ liệu Chứng khoán Việt Nam)
  - **Phụ**: Yahoo Finance (Dự phòng)
  - **Giai đoạn**: 2020-01-01 đến 2025-08-15 (1,366 ngày giao dịch)
  - **VN-Index**: Lấy từ vnstock (VNINDEX)
  - **Lãi suất phi rủi ro**: 2.7% (Trái phiếu Chính phủ Việt Nam)

#### 3. **Dự đoán Tương lai (Future Prediction - 2026-2030)**

- **Mô hình Dự đoán**: Random Forest với 20+ đặc trưng
- **Giai đoạn Dự đoán**: 4 năm (2026-2030)
- **Phương pháp**: Geometric Brownian Motion + Random Forest
- **Kết quả Dự đoán**:
  - **PPC**: Best performer trong tất cả 4 năm
  - **Highest Return**: 21.10% trong năm 2030
  - **Best Sharpe**: 104.07 trong năm 2030
- **Tần suất Best Performer**: PPC (4/4 năm)

#### 4. **Phân tích So sánh**

- So sánh song song cả hai mô hình
- Đánh giá lợi nhuận hàng ngày và tỷ lệ Sharpe
- Đánh giá hiệu suất danh mục

### 🔧 Triển khai Kỹ thuật

#### **Nguồn Dữ liệu**

- **vnstock**: Dữ liệu thị trường chứng khoán Việt Nam (chính)
- **Yahoo Finance**: Nguồn dữ liệu dự phòng
- **Giai đoạn**: 2020-01-01 đến 2025-08-15 (1,365+ ngày giao dịch)

#### **Tính năng Chính**

- **Thu thập dữ liệu thời gian thực** từ nhiều nguồn với cơ chế dự phòng
- **Kỹ thuật đặc trưng toàn diện** với 20+ biến
- **Tối ưu hóa danh mục tiên tiến** sử dụng tối ưu hóa scipy
- **Nhiều đầu ra trực quan** với biểu đồ chất lượng xuất bản
- **Xuất Excel** với kết quả chi tiết qua 15+ sheet
- **Tích hợp VN-Index** với phân tích tương quan

#### **Quản lý Rủi ro**

- Xử lý dữ liệu thiếu với forward-fill và nội suy
- Xử lý lỗi mạnh mẽ cho lỗi nguồn dữ liệu
- Giá trị mặc định dự phòng cho tỷ số tài chính
- Thực thi ràng buộc danh mục (trọng số tổng bằng 1, không bán khống)

### 📈 Kết quả Chính

#### **Hiệu suất Cổ phiếu Cá nhân (Mô hình Định lượng)**

| Cổ phiếu | Lợi nhuận Hàng ngày | Lợi nhuận Hàng năm | Độ biến động | Tỷ lệ Sharpe | Tương quan VN-Index |
| -------- | ------------------- | ------------------ | ------------ | ------------ | ------------------- |
| OIL      | 0.000782            | 9.57%              | 45.74%       | 0.1501       | 0.5895              |
| POW      | 0.000537            | 6.31%              | 38.21%       | 0.0944       | 0.6203              |
| GEG      | 0.000278            | -0.64%             | 39.02%       | -0.0857      | 0.5185              |
| COM      | 0.000461            | -3.76%             | 55.42%       | -0.1165      | 0.1043              |
| PLX      | 0.000093            | -2.92%             | 32.50%       | -0.1728      | 0.6165              |
| PPC      | -0.000082           | -5.47%             | 26.77%       | -0.3052      | 0.5019              |

#### **Hiệu suất Danh mục (Top 3 Cổ phiếu: OIL, POW, GEG)**

| Danh mục              | Lợi nhuận Kỳ vọng | Độ biến động | Tỷ lệ Sharpe | Đặc điểm Chính                             |
| --------------------- | ----------------- | ------------ | ------------ | ------------------------------------------ |
| **Danh mục Tangency** | 8.86%             | 40.36%       | **0.1526**   | Max Sharpe (78.3% OIL, 21.7% POW)          |
| **Danh mục GMV**      | 4.34%             | 32.76%       | 0.0500       | Min Risk (20.9% OIL, 41.0% POW, 38.1% GEG) |
| **Trọng số Đều**      | 5.08%             | 33.15%       | 0.0717       | Cân bằng (33.3% mỗi)                       |

#### **🔮 Dự đoán Tương lai (2026-2030) - Random Forest Model**

| **Cổ phiếu** | **Lợi nhuận Trung bình** | **Tỷ lệ Sharpe Trung bình** | **Best Year** | **Best Sharpe** | **Xu hướng** |
| ------------ | ------------------------ | --------------------------- | ------------- | --------------- | ------------ |
| **PPC**      | **16.94%**               | **65.43**                   | 2030          | 104.07          | 📈 Tăng mạnh |
| **OIL**      | 2.99%                    | 8.95                        | 2030          | 35.71           | 📊 Ổn định   |
| **COM**      | -9.60%                   | -8.79                       | 2030          | -8.36           | 📉 Giảm      |
| **GEG**      | -8.71%                   | -67.83                      | 2029          | -67.37          | 📉 Giảm      |
| **POW**      | -7.30%                   | -37.47                      | 2029          | -37.33          | 📉 Giảm      |
| **PLX**      | -13.55%                  | -648.21                     | 2029          | -595.48         | 📉 Giảm mạnh |

#### **Điểm nổi bật So sánh Mô hình**

| Cổ phiếu | Sharpe Định lượng | Sharpe Random Forest | Cải thiện    | Tương quan VN-Index |
| -------- | ----------------- | -------------------- | ------------ | ------------------- |
| **PPC**  | -0.3052           | **2.9729**           | **+3.28** 🚀 | 0.5019              |
| **OIL**  | 0.1501            | **1.7896**           | **+1.64** 📈 | 0.5895              |
| **POW**  | 0.0944            | **1.7081**           | **+1.61** 📈 | 0.6203              |
| **GEG**  | -0.0857           | **1.2788**           | **+1.36** 📈 | 0.5185              |
| **COM**  | -0.1165           | **0.2560**           | **+0.37** 📊 | 0.1043              |
| PLX      | -0.1728           | -0.7182              | -0.55 📉     | 0.6165              |

### 🎨 Files Đầu ra

#### **Trực quan hóa (16 Biểu đồ Tổng cộng)**

**📊 Biểu đồ Quantitative Flow (6):**

1. **`step1_returns_analysis.png`** - Lợi nhuận Hàng ngày, Hàng tháng, Hàng năm, Tích lũy
2. **`step2_volatility_analysis.png`** - Độ lệch Chuẩn, Phương sai, Độ biến động, Độ lệch Bán
3. **`step3_risk_adjusted_metrics.png`** - Tỷ lệ Sharpe, Rủi ro vs Lợi nhuận, Lợi nhuận Vượt trội
4. **`step4_ew_vs_cw_comparison.png`** - So sánh danh mục trọng số đều vs vốn hóa
5. **`step5_portfolio_weights_analysis.png`** - Phân tích top 3 cổ phiếu và trọng số
6. **`step6_portfolio_optimization.png`** - Kết quả tối ưu hóa danh mục

**🔮 Biểu đồ Dự đoán ML (6):**

7. **`year_2026_prediction.png`** - Dự đoán ML cho năm 2026
8. **`year_2027_prediction.png`** - Dự đoán ML cho năm 2027
9. **`year_2028_prediction.png`** - Dự đoán ML cho năm 2028
10. **`year_2029_prediction.png`** - Dự đoán ML cho năm 2029
11. **`year_2030_prediction.png`** - Dự đoán ML cho năm 2030
12. **`4year_summary_prediction.png`** - Tóm tắt dự đoán 4 năm

**📈 Biểu đồ Kết quả ML (2):**

13. **`ml_daily_returns_6stocks_4years.png`** - Lợi nhuận hàng ngày ML cho tất cả 6 cổ phiếu trong 4 năm
14. **`ml_sharpe_ratios_6stocks_4years.png`** - Tỷ lệ Sharpe ML cho tất cả 6 cổ phiếu trong 4 năm

**🔄 Biểu đồ So sánh Mô hình (2):**

15. **`comparison_quanti_vs_ml_daily_returns.png`** - So sánh lợi nhuận hàng ngày Định lượng vs ML
16. **`comparison_quanti_vs_ml_sharpe_ratios.png`** - So sánh tỷ lệ Sharpe Định lượng vs ML

#### **Xuất Dữ liệu**

- **`portfolio_optimization_complete_results_v3.xlsx`** - Kết quả chi tiết với **15+ sheet toàn diện**:
  - `Individual_Performance`: Chỉ số cấp cổ phiếu
  - `Quantitative_Summary`: 14 chỉ số định lượng cốt lõi cho mỗi cổ phiếu
  - `Correlation_Matrix`: Phân tích tương quan 6x6
  - `Covariance_Matrix`: Phân tích hiệp phương sai 6x6
  - `Portfolio_Performance`: So sánh danh mục
  - `Portfolio_Weights`: Phân bổ tối ưu
  - `Return_Variables`: 5 chỉ số lợi nhuận cho mỗi cổ phiếu
  - `Risk_Variables`: 5 chỉ số rủi ro cho mỗi cổ phiếu
  - `Risk_Adjusted_Variables`: Tỷ lệ Sharpe & RTRR
  - `Portfolio_Variables`: Trọng số và lợi nhuận EW & CW
  - `Additional_Metrics`: VaR, CVaR, Drawdown, Skewness, Kurtosis
  - `Volume_Statistics`: Phân tích khối lượng (triệu)
  - `Financial_Ratios`: ROA, Đòn bẩy, Tỷ số Tiền mặt, Vòng quay Tài sản
  - `VN_Index_Correlation`: Phân tích tương quan thị trường
  - `Model_Comparison`: Định lượng vs Random Forest
  - `Summary`: Phát hiện chính và khuyến nghị

### 🚀 Cách Chạy

#### **Điều kiện Tiên quyết**

```bash
pip install -r requirements.txt
```

**Gói yêu cầu:**

- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy
- yfinance, vnstock
- openpyxl (cho xuất Excel)

#### **Thực thi**

**Phiên bản Mới nhất (Khuyến nghị - Phân tích Định lượng + ML Hoàn chỉnh với 16 Biểu đồ):**

```bash
python quantitative_flow_analysis.py
```

**Phiên bản Trước:**

```bash
# Version 3 (Phân tích Hoàn chỉnh với Tất cả Biến)
python portfolio_optimization_energy_vietnam_v3.py

# Version 2 (Biến Định lượng Hoàn chỉnh)
python portfolio_optimization_energy_vietnam_v2.py

# Version 1 (Gốc)
python portfolio_optimization_energy_vietnam.py
```

**Thời gian chạy dự kiến:** 3-4 phút
**Đầu ra:** **16 file PNG** + đầu ra console toàn diện với:

- 6 biểu đồ Quantitative Flow
- 5 biểu đồ dự đoán ML hàng năm (2026-2030)
- 2 biểu đồ kết quả ML
- 2 biểu đồ so sánh mô hình
- 1 biểu đồ tóm tắt

### 📊 Hiểu biết Chính

#### **🏆 Chiến lược Hiệu suất Tốt nhất**

- **Danh mục Tangency** đạt lợi nhuận điều chỉnh rủi ro cao nhất (Sharpe = 0.1526)
- **Phân bổ nặng vào OIL** (78.3%) thúc đẩy hiệu suất
- **Random Forest vượt trội đáng kể** so với phương pháp truyền thống cho 5/6 cổ phiếu

#### **🔍 Động lực Thị trường**

- **OIL và POW** cho thấy lợi nhuận điều chỉnh rủi ro dương
- **Độ biến động cao** trong lĩnh vực năng lượng (26-55% hàng năm)
- **Phân tích yếu tố** cho thấy EW hơi vượt trội so với danh mục CW
- **Tương quan VN-Index** thay đổi đáng kể: POW (0.6203) > PLX (0.6165) > OIL (0.5895)

#### **🤖 Lợi thế Học máy**

- **Random Forest nắm bắt các mẫu** bị bỏ lỡ bởi phân tích truyền thống
- **Giảm độ biến động** thông qua mô hình rủi ro tốt hơn
- **5/6 cổ phiếu** cho thấy cải thiện tỷ lệ Sharpe với phương pháp ML

#### **🔍 Trường hợp Đặc biệt: Lợi thế Truyền thống của PLX**

**PLX (Petrolimex)** là cổ phiếu duy nhất mà phương pháp định lượng truyền thống vượt trội Random Forest:

| Mô hình           | Lợi nhuận Hàng ngày | Độ lệch Chuẩn Hàng ngày | Tỷ lệ Sharpe | Hiệu suất  |
| ----------------- | ------------------- | ----------------------- | ------------ | ---------- |
| **Định lượng**    | +0.000093           | 0.020554                | **-0.1728**  | ✅ Tốt hơn |
| **Random Forest** | -0.000045           | 0.003360                | **-0.7182**  | ❌ Tệ hơn  |

**Tại sao PLX cho thấy Lợi thế Truyền thống:**

- **Mô hình ML dự đoán lợi nhuận âm** (-0.000045) vs dương truyền thống (+0.000093)
- **Giảm độ biến động (84%)** không đủ để bù đắp dự đoán lợi nhuận âm
- **Các mẫu cụ thể của công ty phức tạp** mà ML không thể nắm bắt hiệu quả
- **Động lực thị trường độc đáo của Petrolimex** có thể yêu cầu phân tích cơ bản truyền thống

**Hiểu biết Chính**: Điều này chứng minh tầm quan trọng của **phương pháp mô hình kép** - không phải tất cả cổ phiếu đều được hưởng lợi như nhau từ ML, và phương pháp truyền thống vẫn giữ giá trị cho một số mẫu thị trường nhất định.

### 🎯 Khuyến nghị Đầu tư

1. **Tập trung vào những người thực hiện tốt nhất**: OIL và POW như nắm giữ cốt lõi
2. **Xem xét Danh mục Tangency**: Phân bổ điều chỉnh rủi ro tối ưu
3. **Tận dụng hiểu biết ML**: Random Forest cung cấp hồ sơ rủi ro-lợi nhuận vượt trội cho 5/6 cổ phiếu
4. **Theo dõi PPC chặt chẽ**: Tiềm năng cải thiện ML cao nhất (+3.28 Sharpe)
5. **Đa dạng hóa trong năng lượng**: Duy trì tiếp xúc qua các tiểu lĩnh vực
6. **Phương pháp lai cho PLX**: Sử dụng phương pháp định lượng truyền thống cho phân tích Petrolimex
7. **Nhận thức tương quan thị trường**: POW cho thấy tương quan VN-Index cao nhất (0.6203)

### 📁 Cấu trúc Dự án

```
ML/
├── quantitative_flow_analysis.py                # 🆕 MỚI NHẤT - Phân tích Định lượng + ML Hoàn chỉnh (16 biểu đồ)
├── portfolio_optimization_energy_vietnam_v3.py  # VERSION 3 - Phân tích hoàn chỉnh với tất cả biến
├── portfolio_optimization_energy_vietnam_v2.py  # VERSION 2 - Biến định lượng hoàn chỉnh
├── portfolio_optimization_energy_vietnam.py     # Script phân tích gốc
├── requirements.txt                              # Dependencies
├── README.md                                    # File này
├── FINAL_SUMMARY.md                             # Tóm tắt điều hành
├── GIỚI THIỆU ĐỀ TÀI.pdf                        # Giới thiệu dự án
├── requirements.pdf                              # Tài liệu yêu cầu
├── step1_returns_analysis.png                   # Phân tích lợi nhuận
├── step2_volatility_analysis.png                # Phân tích độ biến động
├── step3_risk_adjusted_metrics.png              # Chỉ số điều chỉnh rủi ro
├── step4_ew_vs_cw_comparison.png                # So sánh danh mục
├── step5_portfolio_weights_analysis.png         # Trọng số danh mục
├── step6_portfolio_optimization.png             # Tối ưu hóa danh mục
├── year_2026_prediction.png                     # Dự đoán 2026
├── year_2027_prediction.png                     # Dự đoán 2027
├── year_2028_prediction.png                     # Dự đoán 2028
├── year_2029_prediction.png                     # Dự đoán 2029
├── year_2030_prediction.png                     # Dự đoán 2030
├── 4year_summary_prediction.png                 # Tóm tắt dự đoán 4 năm
├── ml_daily_returns_6stocks_4years.png          # Lợi nhuận hàng ngày ML
├── ml_sharpe_ratios_6stocks_4years.png          # Tỷ lệ Sharpe ML
├── comparison_quanti_vs_ml_daily_returns.png    # So sánh mô hình - lợi nhuận
├── comparison_quanti_vs_ml_sharpe_ratios.png    # So sánh mô hình - Sharpe
└── portfolio_optimization_complete_results_v3.xlsx # Kết quả chi tiết (15+ sheet)
```

### 🔧 Ghi chú Kỹ thuật

- **Xử lý dữ liệu**: Xử lý lỗi mạnh mẽ với nhiều cơ chế dự phòng
- **Kỹ thuật đặc trưng**: 20+ biến bao gồm đặc trưng kỹ thuật, cơ bản và thời gian
- **Tối ưu hóa**: Tối ưu hóa danh mục dựa trên scipy với ràng buộc thực tế
- **Xác thực**: Kiểm tra chéo và chỉ số hiệu suất cho mô hình ML
- **Xuất**: Đầu ra chất lượng chuyên nghiệp phù hợp cho trình bày
- **Tích hợp VN-Index**: Phân tích tương quan thị trường thời gian thực
- **Màu sắc đồng nhất**: Tất cả biểu đồ sử dụng màu chủ đạo #060270

### 📞 Hỗ trợ

Đối với câu hỏi về triển khai hoặc giải thích kết quả, vui lòng tham khảo đầu ra console chi tiết và file xuất Excel chứa giải thích toàn diện và kết quả số.

---

**Giai đoạn Phân tích**: 2020-2025 | **Cập nhật Cuối**: Tháng 9/2025 | **Khung**: Định lượng + Học máy + Dự đoán Tương lai | **Phiên bản**: 4.0 (Phân tích Hoàn chỉnh với Tất cả Biến + Dự đoán Tương lai 4 Năm)
