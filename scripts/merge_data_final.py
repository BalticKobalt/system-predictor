import pandas as pd
import os

DATA_PATH = 'data/'

files = {
    'gold': 'gold_prices.csv',
    'brent': 'brent_prices.csv',
    'vix': 'vix_prices.csv',
    'dxy': 'dxy_prices.csv',
}

def load_yfinance_csv(filename):
    filepath = os.path.join(DATA_PATH, filename)
    print(f"Пытаюсь загрузить: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath, 
                         skiprows=2,
                         parse_dates=[0],
                         header=None,
                         names=['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume'])
        
        df = df.set_index('Date')
        df = df[['Close']]
        
        print(f"✅ Загружен: {filename}, дат: {len(df)}")
        return df
        
    except Exception as e:
        print(f"❌ Ошибка загрузки {filename}: {e}")
        return None

loaded_dfs = {}
for name, file in files.items():
    df = load_yfinance_csv(file)
    if df is not None:
        loaded_dfs[name] = df

if not loaded_dfs:
    print("❌ Не удалось загрузить ни одного файла.")
    exit()

print("\nОбъединяем данные...")
first_key = list(loaded_dfs.keys())[0]
main_df = loaded_dfs[first_key]
print(f"Основа: {first_key}")

for name, df in loaded_dfs.items():
    if name == first_key:
        continue
    series = df['Close'].rename(name)
    main_df = pd.merge(main_df, series, how='outer', left_index=True, right_index=True)
    print(f"➕ Добавлен: {name}")

# Исправленный способ заполнения пропусков
main_df = main_df.ffill()  # вместо fillna(method='ffill')
main_df = main_df.dropna(how='all')

main_df.to_csv('data/master_data.csv')
print(f"\n✅ Готовая таблица сохранена!")
print(f"   Размер: {main_df.shape[0]} строк, {main_df.shape[1]} колонок")
print(f"   Колонки: {main_df.columns.tolist()}")
print(f"   Период: с {main_df.index.min()} по {main_df.index.max()}")