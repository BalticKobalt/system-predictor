import pandas as pd
import os

print("📊 Добавляем GPR и GECON в master_data.csv...")

# 1. Загружаем базу
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)
print(f"📊 База: {len(df)} строк, колонки: {df.columns.tolist()}")

# 2. Добавляем AI-GPR (ежедневный)
ai_gpr = pd.read_csv('data/ai_gpr_data_daily.csv', parse_dates=['Date'], index_col='Date')
df['ai_gpr'] = ai_gpr['GPR_AI']
print("✅ Добавлен AI-GPR")

# 3. Добавляем GPR (из .dta)
gpr = pd.read_stata('data/data_gpr_export.dta')
gpr['month'] = pd.to_datetime(gpr['month'])
gpr = gpr.set_index('month')
df['gpr'] = gpr['GPR']
print("✅ Добавлен GPR")

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
print("✅ Добавлен GECON")

# 5. Заполняем пропуски
df = df.ffill().bfill().fillna(0)

# 6. Сохраняем
df.to_csv('data/master_data.csv')
print(f"\n✅ master_data.csv обновлён!")
print(f"📊 {len(df)} строк, колонки: {df.columns.tolist()}")