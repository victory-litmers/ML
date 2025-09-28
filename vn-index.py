from vnstock import Vnstock
import pandas as pd

stock = Vnstock().stock(symbol='PLX', source='VCI')

ratios = stock.finance.ratio(period='quarter', lang='vi')

# Flatten MultiIndex columns để có thể export Excel
ratios_flat = ratios.copy()
if isinstance(ratios_flat.columns, pd.MultiIndex):
    ratios_flat.columns = ['_'.join(col).strip() for col in ratios_flat.columns.values]

# Export to Excel
ratios_flat.to_excel('plx_ratios.xlsx', index=True)

print("📊 PLX Financial Ratios:")
print(f"Shape: {ratios.shape}")
print(f"Columns: {ratios.columns.tolist()[:5]}...")  # Show first 5 columns
print("\n📊 Latest Quarter Data:")
print(ratios.iloc[0])  # Latest quarter