from vnstock import Vnstock, Quote


def get_stock_data(ticker, start_date='2020-01-01', end_date='2025-08-15'):
    """Lấy dữ liệu cổ phiếu từ vnstock"""
    try:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        data = stock.quote.history(symbol=ticker, start=start_date, end=end_date, interval='1D')
        if data is not None and not data.empty:
            return data
        else:
            print(f"❌ {ticker}: Không có dữ liệu")
            return None
    except Exception as e:
        print(f"❌ {ticker}: Lỗi {e}")
        return None

tickers = ['PLX', 'OIL', 'GAS', 'PPC', 'GEG', 'POW']
stock_data = {}
for ticker in tickers:
    print(f"   📈 {ticker}...")
    data = get_stock_data(ticker)
    if data is not None:
        stock_data[ticker] = data
        print(f"   ✅ {ticker}: {len(data)} ngày giao dịch")
    else:
        print(f"   ❌ {ticker}: Không có dữ liệu")

print(stock_data)