import pandas as pd
import os

print("🔄 Восстанавливаем master_data.csv из исходных файлов...")

# 1. Загружаем основные финансовые данные
df = pd.read_csv('data/gold_prices.csv', parse_dates=['Date'], index_col='Date')
df['brent'] = pd.read_csv('data/brent_prices.csv', parse_dates=['Date'], index_col='Date')['Close']
df['vix'] = pd.read_csv('data/vix_prices.csv', parse_dates=['Date'], index_col='Date')['Close']
df['dxy'] = pd.read_csv('data/dxy_prices.csv', parse_dates=['Date'], index_col='Date')['Close']

# 2. Добавляем AI-GPR (ежедневный)
ai_gpr = pd.read_csv('data/ai_gpr_data_daily.csv', parse_dates=['Date'], index_col='Date')
df['ai_gpr'] = ai_gpr['GPR_AI']

# 3. Добавляем GPR (из .dta)
gpr = pd.read_stata('data/data_gpr_export.dta')
gpr['month'] = pd.to_datetime(gpr['month'])
gpr = gpr.set_index('month')
df['gpr'] = gpr['GPR']

# 4. Добавляем GECON
gecon = pd.read_excel('data/GECON_indicator.xlsx', engine='openpyxl')
def parse_gecon_date(date_str):
    if isinstance(date_str, str) and 'M' in date_str:
        year, month = date_str.split('M')
        return pd.Timestamp(f"{year}-{month.zfill(2)}-01")
    return pd.NaT
gecon['parsed_date'] = gecon['Unnamed: 0'].apply(parse_gecon_date)
gecon = gecon.set_index('parsed_date')
df['gecon'] = gecon['Global Economic Conditions Indicator']

# 5. Заполняем пропуски
df = df.ffill().bfill().fillna(0)

# 6. Сохраняем
df.to_csv('data/master_data.csv')
print(f"✅ master_data.csv восстановлен! {len(df)} строк, с {df.index.min()} по {df.index.max()}")
print(f"📋 Колонки: {df.columns.tolist()}")