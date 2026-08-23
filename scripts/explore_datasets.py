import pandas as pd

print("🔍 Изучаем датасеты...")

# 1. Загружаем кризисы
crisis = pd.read_excel('data/global_crisis_data.xlsx', engine='openpyxl')
print("\n📊 КРИЗИСЫ:")
print(f"Колонки: {crisis.columns.tolist()}")
print(f"Первые 5 строк:\n{crisis.head()}")
print(f"Размер: {crisis.shape}")

# 2. Загружаем конфликты
conflicts = pd.read_csv('data/UcdpPrioConflict_v26_1.csv')
print("\n📊 КОНФЛИКТЫ:")
print(f"Колонки: {conflicts.columns.tolist()}")
print(f"Первые 5 строк:\n{conflicts.head()}")
print(f"Размер: {conflicts.shape}")