import pandas as pd
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline
import torch
import numpy as np

print("📊 Загружаем данные...")
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# Выбираем колонки для прогноза
features = ['Close', 'brent', 'vix', 'dxy', 'ai_gpr']

# Удаляем строки, где есть пропуски хотя бы в одной колонке
df_selected = df[features].dropna()

print(f"📈 Данные: {len(df_selected)} дней, с {df_selected.index.min()} по {df_selected.index.max()}")
print(f"📋 Используемые признаки: {features}")

if len(df_selected) == 0:
    print("❌ Нет данных после удаления пропусков. Попробуй другие колонки или заполни пропуски.")
    exit()

# Берем только золото для прогноза (Chronos 1 не умеет многомерный прогноз легко)
# Но мы можем использовать все данные для обучения, а прогноз делать только для золота
context = torch.tensor(df_selected['Close'].values, dtype=torch.float32).unsqueeze(0)

print("🧠 Загружаем модель Chronos...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
)

print("🔮 Делаем прогноз на 30 дней...")
# Правильный синтаксис для predict_quantiles
quantiles, mean = pipeline.predict_quantiles(
    context,
    prediction_length=30,
    quantile_levels=[0.1, 0.5, 0.9]
)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(df_selected.index, df_selected['Close'], label="Исторические данные", color='blue')

forecast_index = range(len(df_selected), len(df_selected) + 30)
low = quantiles[0, :, 0].detach().numpy()
median = quantiles[0, :, 1].detach().numpy()
high = quantiles[0, :, 2].detach().numpy()

plt.plot(forecast_index, median, label="Прогноз (медиана)", color='red', linewidth=2)
plt.fill_between(forecast_index, low, high, color='red', alpha=0.2, label="Доверительный интервал (10%-90%)")
plt.title("Прогноз цены золота с использованием всех доступных данных")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_multivariate.png')
print("✅ График сохранён: data/forecast_multivariate.png")
plt.show()