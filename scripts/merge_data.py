import pandas as pd
import os

# Указываем путь к папке с данными
DATA_PATH = '../data/'

# 1. Загружаем все файлы в словарь
files = {
    'gold': 'gold_prices.csv',
    'brent': 'brent_prices.csv',
    'vix': 'vix_prices.csv',
    'dxy': 'dxy_prices.csv',
    'gpr': 'data_gpr_export.xls',  # Это может быть .xls, для него нужен другой движок
    # Добавь сюда остальные файлы по мере необходимости
}

def load_data(filename):
    """Универсальная загрузка для CSV и Excel"""
    filepath = os.path.join(DATA_PATH, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filename.endswith('.xls') or filename.endswith('.xlsx'):
        # Для файлов GPR понадобится библиотека openpyxl
        df = pd.read_excel(filepath, engine='openpyxl') 
    else:
        raise ValueError(f"Неподдерживаемый формат: {filename}")
    
    # Приводим колонку с датой к единому формату
    # Название колонки может отличаться, например, 'Date' или 'date'
    date_col = 'Date' if 'Date' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col])
    # Устанавливаем дату как индекс для удобства объединения
    df = df.set_index(date_col)
    return df

# 2. Загружаем все датафреймы
loaded_dfs = {}
for name, file in files.items():
    try:
        loaded_dfs[name] = load_data(file)
        print(f"✅ Загружен: {file}")
    except Exception as e:
        print(f"❌ Ошибка загрузки {file}: {e}")

# 3. Объединяем их в один датафрейм
# Первый датафрейм берем как основу
main_df = loaded_dfs['gold']
for name, df in loaded_dfs.items():
    if name == 'gold':
        continue
    # Объединяем по индексу (датам) с помощью outer join 
    # Это сохранит все даты из всех файлов
    main_df = pd.merge(main_df, df, how='outer', left_index=True, right_index=True)

# 4. Заполняем пропуски (NaN) 
# Для ежемесячных данных (как GECON) логично использовать forward fill
main_df = main_df.fillna(method='ffill')

# 5. Сохраняем готовую таблицу
main_df.to_csv('../data/master_data.csv')
print(f"✅ Готовая таблица сохранена! Размер: {main_df.shape}")