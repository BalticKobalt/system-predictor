import pandas as pd
import os

print("📊 Обновляем master_data.csv со всеми колонками...")

# 1. Загружаем базовый файл
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Добавляем GECON
gecon_file = 'data/GECON_indicator.xlsx'
if os.path.exists(gecon_file):
    gecon_df = pd.read_excel(gecon_file, engine='openpyxl')
    gecon_df['Unnamed: 0'] = pd.to_datetime(gecon_df['Unnamed: 0'].str.replace('M', '-') + '-01')
    gecon_df = gecon_df.set_index('Unnamed: 0')
    df['gecon'] = gecon_df['Global Economic Conditions Indicator']
    print("✅ Добавлен GECON")

# 3. Добавляем GPR (из .dta)
dta_file = 'data/data_gpr_export.dta'
if os.path.exists(dta_file):
    gpr_df = pd.read_stata(dta_file)
    gpr_df['month'] = pd.to_datetime(gpr_df['month'])
    gpr_df = gpr_df.set_index('month')
    df['gpr'] = gpr_df['GPR']
    print("✅ Добавлен GPR")

# 4. Сохраняем
df.to_csv('data/master_data.csv')
print("✅ master_data.csv обновлён!")
print(f"📋 Колонки: {df.columns.tolist()}")
print(f"📈 Данные: {len(df)} строк, с {df.index.min()} по {df.index.max()}")