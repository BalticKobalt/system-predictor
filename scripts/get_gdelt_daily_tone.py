import gdelt
import pandas as pd

print("📰 Скачиваем ежедневную тональность GDELT...")

# Инициализируем GDELT
gd = gdelt.gdelt(version=2)

# Скачиваем готовый ежедневный дайджест (не все события, а агрегированные данные)
# Это делается через метод `Daily` (а не Search)
try:
    # Скачиваем данные за последние 30 дней (для теста)
    daily_data = gd.Daily('2025-07-01', '2025-07-31')
    
    # Если данных нет, попробуем другой период
    if daily_data is None or daily_data.empty:
        print("⚠️ Нет данных за июль 2025, пробуем август 2025...")
        daily_data = gd.Daily('2025-08-01', '2025-08-21')
    
    print(f"✅ Скачано {len(daily_data)} записей")
    print(daily_data.head())
    
    # Сохраняем
    daily_data.to_csv('data/gdelt_daily_tone.csv', index=False)
    print("✅ Файл сохранён: data/gdelt_daily_tone.csv")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")