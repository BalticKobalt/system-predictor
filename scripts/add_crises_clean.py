import pandas as pd
import numpy as np

print("🔧 Добавляем кризисы и конфликты в master_data.csv...")

# 1. Загружаем мастер-таблицу (из копии)
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# Проверяем, что данные загрузились
print(f"📊 Мастер-таблица: {len(df)} строк, с {df.index.min()} по {df.index.max()}")

if len(df) == 0:
    print("❌ Нет данных! Проверь, что файл не пустой.")
    exit()

# 2. Обработка кризисов (Harvard)
print("\n📊 Обрабатываем кризисы...")
crisis = pd.read_excel('data/global_crisis_data.xlsx', engine='openpyxl')

# Создаём флаг кризиса
crisis['crisis_flag'] = 0
for col in ['Banking Crisis ', 'Systemic Crisis', 'Currency Crises', 'Inflation Crises']:
    if col in crisis.columns:
        crisis['crisis_flag'] = crisis['crisis_flag'] | crisis[col].notna()

# Группируем по году
yearly_crisis = crisis.groupby('Year')['crisis_flag'].mean().reset_index()
yearly_crisis.columns = ['year', 'crisis_ratio']

# Добавляем в мастер-таблицу
df['year'] = df.index.year
df = df.join(yearly_crisis.set_index('year'), on='year', how='left')
df['crisis_ratio'] = df['crisis_ratio'].ffill().fillna(0)
df = df.drop(columns=['year'])

print("✅ Добавлен признак 'crisis_ratio'")

# 3. Обработка конфликтов (UCDP)
print("\n📊 Обрабатываем конфликты...")
conflicts = pd.read_csv('data/UcdpPrioConflict_v26_1.csv')

# Преобразуем даты
conflicts['start_date'] = pd.to_datetime(conflicts['start_date'], errors='coerce')
conflicts['start_year'] = conflicts['start_date'].dt.year

# Группируем по году
yearly_intensity = conflicts.groupby('start_year')['intensity_level'].sum().reset_index()
yearly_intensity.columns = ['year', 'conflict_intensity']

# Добавляем в мастер-таблицу
df['year'] = df.index.year
df = df.join(yearly_intensity.set_index('year'), on='year', how='left')
df['conflict_intensity'] = df['conflict_intensity'].ffill().fillna(0)
df = df.drop(columns=['year'])

print("✅ Добавлен признак 'conflict_intensity'")

# 4. Сохраняем
df.to_csv('data/master_data.csv')
print("\n✅ master_data.csv обновлён!")
print(f"📋 Итоговые колонки: {df.columns.tolist()}")
print(f"📊 Данные: с {df.index.min()} по {df.index.max()}, {len(df)} строк")