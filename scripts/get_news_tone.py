import pandas as pd
from gdeltdoc import GdeltDoc, Filters

print("📰 Скачиваем новостную тональность из GDELT...")

# 1. Настройка фильтра
# Ищем ключевые слова, связанные с геополитикой и экономикой
f = Filters(
    keyword="war OR conflict OR economy OR oil OR gold OR crisis",
    start_date="2025-01-01",   # можно поменять на нужный период
    end_date="2026-08-21",     # сегодняшняя дата
    country=["US", "GB", "FR", "DE", "RU", "CN"],  # основные страны
    language="english"
)

# 2. Создаём клиент GDELT
gd = GdeltDoc()

print("🔍 Запрашиваем данные...")

try:
    # 3. Получаем таймлайн тональности (средний тон по дням)
    timeline = gd.timeline_search("timelinetone", f)
    
    # 4. Сохраняем результат
    timeline.to_csv('data/news_tone_timeline.csv')
    print(f"✅ Таймлайн тональности сохранён: {len(timeline)} дней")
    print(timeline.head())
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("Попробуй изменить даты или ключевые слова.")