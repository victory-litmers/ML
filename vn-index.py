from vnstock import stock_historical_data

df = stock_historical_data("VNINDEX", "2000-06-01", "2023-12-29", "1", "index", source='TCBS')

print(df)