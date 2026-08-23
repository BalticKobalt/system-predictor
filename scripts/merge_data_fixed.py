import pandas as pd
import os

# Указываем путь к папке с данными
DATA_PATH = 'data/'

# Список файлов для загрузки
files = {
    'gold': 'gold_prices.csv',
    'brent': 'brent_prices.csv',
    'vix': 'vix_prices.csv',
    'dxy': 'dxy_prices.csv',
    # GPR пока пропускаем, т.к. его нет в папке data
}

def load_yfinance_csv(filename):
    """Специальная загрузка для CSV от yfinance (дата — это индекс)"""
    filepath = os.path.join(DATA_PATH, filename)
    print(f"Пытаюсь загрузить: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath}")
        return None
    
    # Читаем CSV с указанием, что первый столбец — это дата (индекс)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Проверяем, что индекс действительно дата
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"⚠️ Индекс не является датой. Пробуем преобразовать...")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"❌ Не удалось преобразовать индекс в дату: {e}")
            return None
    
    print(f"✅ Загружен: {filename}, дат: {len(df)}")
    return df

# Загружаем все файлы
loaded_dfs = {}
for name, file in files.items():
    df = load_yfinance_csv(file)
    if df is not None:
        loaded_dfs[name] = df

if not loaded_dfs:
    print("❌ Не удалось загрузить ни одного файла.")
    exit()

# Объединяем все датафреймы по датам (индексу)
print("\nОбъединяем данные...")
first_key = list(loaded_dfs.keys())[0]
main_df = loaded_dfs[first_key]
print(f"Основа: {first_key}")

for name, df in loaded_dfs.items():
    if name == first_key:
        continue
    # Берём только колонку 'Close' (цену закрытия) из каждого файла
    # и переименовываем её в название актива
    col_name = name  # gold, brent, vix, dxy
    if 'Close' in df.columns:
        series = df['Close'].rename(col_name)
        main_df = pd.merge(main_df, series, how='outer', left_index=True, right_index=True)
        print(f"➕ Добавлен: {name} (колонка Close)")
    else:
        print(f"⚠️ В {name} нет колонки Close")

# Заполняем пропуски (forward fill)
main_df = main_df.fillna(method='ffill')

# Убираем строки, где все значения NaN (если есть)
main_df = main_df.dropna(how='all')

# Сохраняем результат
main_df.to_csv('data/master_data.csv')
print(f"\n✅ Готовая таблица сохранена!")
print(f"   Размер: {main_df.shape[0]} строк, {main_df.shape[1]} колонок")
print(f"   Колонки: {main_df.columns.tolist()}")
print(f"   Период: с {main_df.index.min()} по {main_df.index.max()}")