import pandas as pd
import os

print("🔄 Собираем master_data.csv заново из исходников...")

# Функция загрузки для yfinance-файлов
def load_yfinance_csv(filename):
    filepath = os.path.join('data', filename)
    print(f"Загружаем {filename}...")
    try:
        # Пропускаем 3 строки: тикер, заголовок, и строку с 'Date'
        df = pd.read_csv(filepath, skiprows=3, 
                         names=['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date')
        return df[['Close']].rename(columns={'Close': filename.replace('_prices.csv', '')})
    except Exception as e:
        print(f"❌ Ошибка загрузки {filename}: {e}")
        return None

# Загружаем все файлы
files = ['gold_prices.csv', 'brent_prices.csv', 'vix_prices.csv', 'dxy_prices.csv']
dfs = {}

for file in files:
    df = load_yfinance_csv(file)
    if df is not None:
        dfs[file] = df

# Объединяем
if not dfs:
    print("❌ Не удалось загрузить ни одного файла!")
    exit()

df = dfs[files[0]]
for file in files[1:]:
    if file in dfs:
        df = df.join(dfs[file], how='outer')

# Заполняем пропуски
df = df.ffill().bfill().fillna(0)

# Сохраняем
df.to_csv('data/master_data.csv')
print(f"\n✅ master_data.csv создан!")
print(f"📊 {len(df)} строк, с {df.index.min()} по {df.index.max()}")
print(f"📋 Колонки: {df.columns.tolist()}")