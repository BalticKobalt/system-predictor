import pandas as pd

# Проверяем структуру каждого CSV-файла
files = ['gold_prices.csv', 'brent_prices.csv', 'vix_prices.csv', 'dxy_prices.csv']

for file in files:
    try:
        df = pd.read_csv(f'data/{file}')
        print(f"\n📁 {file}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"Первые 3 строки:\n{df.head(3)}")
        print(f"Индекс: {df.index}")
    except Exception as e:
        print(f"❌ Ошибка при чтении {file}: {e}")