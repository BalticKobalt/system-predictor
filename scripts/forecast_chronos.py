import pandas as pd
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline

print("📊 Загружаем данные...")

# 1. Загружаем данные
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Готовим DataFrame для Chronos в правильном формате
# Убеждаемся, что данные чистые
series = df['Close'].dropna()

# Создаём DataFrame в формате, который ожидает Chronos:
# - id: идентификатор ряда (для одного ряда ставим 'gold')
# - timestamp: дата в правильном формате datetime
# - target: значение
series_df = pd.DataFrame({
    'id': 'gold',
    'timestamp': series.index,  # уже datetime
    'target': series.values
})

print(f"📈 Данные: {len(series)} дней, с {series.index.min()} по {series.index.max()}")
print(f"📋 Формат данных: {series_df.head(2)}")

# 3. Загружаем модель
print("🧠 Загружаем модель Chronos...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
)

# 4. Делаем прогноз через predict_df (официальный метод)
print("🔮 Делаем прогноз на 30 дней...")
pred_df = pipeline.predict_df(
    series_df,
    prediction_length=30,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

# 5. Визуализация
plt.figure(figsize=(12, 6))

# Исторические данные
plt.plot(series_df['timestamp'], series_df['target'], label="Исторические данные", color='blue')

# Прогноз
gold_forecast = pred_df[pred_df['id'] == 'gold']
plt.plot(gold_forecast['timestamp'], gold_forecast['0.5'], 
         label="Прогноз (медиана)", color='red', linewidth=2)

# Доверительный интервал
plt.fill_between(
    gold_forecast['timestamp'],
    gold_forecast['0.1'],
    gold_forecast['0.9'],
    color='red', alpha=0.2, label="Доверительный интервал (10%-90%)"
)

plt.title("Прогноз цены золота с помощью Chronos")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_gold.png')
print("✅ График сохранён: data/forecast_gold.png")
plt.show()