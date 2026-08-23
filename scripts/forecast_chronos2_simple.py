import pandas as pd
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline
import torch

print("📊 Загружаем данные...")

# 1. Загружаем данные
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Берем колонку с золотом
series = df['Close'].dropna()

# 3. Удаляем все строки, где индекс — это строка (например, 'Date')
# Проверяем, является ли каждый элемент индекса строкой
series = series[~series.index.astype(str).str.contains('Date', case=False)]

# 4. Удаляем NaT из индекса (если остались)
series = series[~series.index.isna()]

# 5. Убеждаемся, что индекс — это datetime
series.index = pd.to_datetime(series.index, errors='coerce')
series = series.dropna()

print(f"📈 Данные: {len(series)} дней, с {series.index.min()} по {series.index.max()}")
print("Последние 5 дат:", series.index[-5:])

# 6. Преобразуем в правильный 3D формат
context = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(f"📐 Форма тензора: {context.shape}")

print("🧠 Загружаем модель Chronos-2...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)

print("🔮 Делаем прогноз на 30 дней...")
quantiles, mean = pipeline.predict_quantiles(
    context,
    prediction_length=30,
    quantile_levels=[0.1, 0.5, 0.9]
)

# 7. Извлекаем прогноз
quantiles_tensor = quantiles[0]

low = quantiles_tensor[0, :, 0].detach().numpy()
median = quantiles_tensor[0, :, 1].detach().numpy()
high = quantiles_tensor[0, :, 2].detach().numpy()

# 8. Визуализация
plt.figure(figsize=(12, 6))

# Исторические данные
plt.plot(series.index, series.values, label="Исторические данные", color='blue')

# Создаём даты для прогноза
last_date = series.index[-1]
prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Прогноз
plt.plot(prediction_dates, median, label="Прогноз (медиана)", color='red', linewidth=2)
plt.fill_between(
    prediction_dates,
    low,
    high,
    color='red', alpha=0.2, label="Доверительный интервал (10%-90%)"
)

plt.title("Прогноз цены золота с Chronos-2 (финальная версия)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_chronos2_gold_final.png')
print("✅ График сохранён: data/forecast_chronos2_gold_final.png")
plt.show()

print("\n📊 Прогноз цены золота на 30 дней (медиана):")
for i, val in enumerate(median):
    print(f"  День {i+1}: {val:.2f}")