import pandas as pd
import os
import re

print("📊 Добавляем GECON в master_data.csv...")

# 1. Загружаем master_data
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Загружаем GECON
gecon_file = 'data/GECON_indicator.xlsx'

if os.path.exists(gecon_file):
    print(f"✅ Найден файл: {gecon_file}")
    
    # Читаем Excel
    gecon_df = pd.read_excel(gecon_file, engine='openpyxl')
    
    print(f"📋 Колонки в GECON: {gecon_df.columns.tolist()}")
    print(f"📊 Первые 5 строк:\n{gecon_df.head(5)}")
    print(f"📊 Последние 5 строк:\n{gecon_df.tail(5)}")
    
    # Колонка с датой
    date_col = 'Unnamed: 0'
    
    # Функция парсинга дат в формате '1973M2' или '2026M7'
    def parse_gecon_date(date_str):
        if isinstance(date_str, str):
            # Убираем всё, кроме цифр и M
            match = re.match(r'(\d{4})M(\d{1,2})', date_str)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                return pd.Timestamp(f"{year}-{month:02d}-01")
        return pd.NaT
    
    # Применяем парсинг
    gecon_df['parsed_date'] = gecon_df[date_col].apply(parse_gecon_date)
    
    # Проверяем, сколько дат распарсилось
    valid_dates = gecon_df['parsed_date'].notna().sum()
    print(f"📅 Распаршено дат: {valid_dates} из {len(gecon_df)}")
    
    # Удаляем строки с NaT
    gecon_df = gecon_df[gecon_df['parsed_date'].notna()]
    gecon_df = gecon_df.set_index('parsed_date')
    
    if len(gecon_df) == 0:
        print("❌ Не удалось распарсить ни одной даты!")
        print("   Проверь формат дат в колонке 'Unnamed: 0'")
        print(f"   Примеры: {gecon_df[date_col].head(10).tolist()}")
        exit()
    
    # Ищем колонку с GECON
    # Приоритет: Global Economic Conditions Indicator (оригинальный)
    gecon_col = None
    for col in gecon_df.columns:
        if 'Global Economic Conditions Indicator' in col and 'Standardized' not in col:
            gecon_col = col
            break
    
    if gecon_col is None:
        for col in gecon_df.columns:
            if 'Standardized GECON' in col and '3' not in col:
                gecon_col = col
                break
    
    if gecon_col is None:
        # Берём первую числовую колонку (кроме даты)
        num_cols = gecon_df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            gecon_col = num_cols[0]
            print(f"⚠️ Использую колонку {gecon_col} как GECON")
    
    if gecon_col is not None:
        # Добавляем в master_data
        df['gecon'] = gecon_df[gecon_col]
        df.to_csv('data/master_data.csv')
        print(f"✅ GECON добавлен в master_data.csv")
        print(f"   Колонка: {gecon_col}")
        print(f"   Период: с {gecon_df.index.min()} по {gecon_df.index.max()}")
        print(f"   Количество записей: {len(gecon_df)}")
    else:
        print("❌ Не найдена колонка с GECON")
else:
    print(f"❌ Файл не найден: {gecon_file}")