import pandas as pd
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline
import torch
import numpy as np

print("📊 Загружаем данные...")

# Загружаем данные
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# Берём колонку с золотом и удаляем пустые значения
series = df['Close'].dropna()
print(f"📈 Данные: {len(series)} дней, с {series.index.min()} по {series.index.max()}")

# Преобразуем данные в тензор PyTorch (как в примерах Chronos)
context = torch.tensor(series.values, dtype=torch.float32)

print("🧠 Загружаем модель Chronos...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
)

print("🔮 Делаем прогноз на 30 дней...")
# Используем правильный синтаксис с позиционным аргументом
quantiles, mean = pipeline.predict_quantiles(
    context,
    prediction_length=30,
    quantile_levels=[0.1, 0.5, 0.9]
)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(series.index, series.values, label="Исторические данные", color='blue')

# Создаём индексы для прогноза (как в официальных примерах)
forecast_index = range(len(series), len(series) + 30)

# Извлекаем прогнозы
low = quantiles[0, :, 0].detach().numpy()
median = quantiles[0, :, 1].detach().numpy()
high = quantiles[0, :, 2].detach().numpy()

plt.plot(forecast_index, median, label="Прогноз (медиана)", color='red', linewidth=2)
plt.fill_between(
    forecast_index,
    low,
    high,
    color='red', alpha=0.2, label="Доверительный интервал (10%-90%)"
)

plt.title("Прогноз цены золота с помощью Chronos")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_gold.png')
print("✅ График сохранён: data/forecast_gold.png")
plt.show()