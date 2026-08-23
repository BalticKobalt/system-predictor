import gdelt
import pandas as pd

# Инициализация (версия 2 — самая свежая)
gd = gdelt.gdelt(version=2)

# Параметры запроса: события с участием России, за 2025 год
events = gd.Search('RUS', 
                   start_date='2025-01-01', 
                   end_date='2025-12-31')

# Сохраняем результат
events.to_csv('B:/PtFiles/data/gdelt_russia_2025.csv', index=False)
print(f"Скачано {len(events)} событий")