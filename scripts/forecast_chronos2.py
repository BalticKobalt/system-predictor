import pandas as pd
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

# 1. Загрузка данных с гарантированным созданием колонок
print("📊 Загружаем данные...")
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Выбираем нужные колонки и СБРАСЫВАЕМ индекс в колонку
features = ['Close', 'brent', 'vix', 'dxy', 'ai_gpr']
df_selected = df[features].copy().reset_index()
df_selected = df_selected.rename(columns={'index': 'timestamp'})

# 3. Заполняем пропуски (безопасный способ для pandas 3.0)
for col in features:
    df_selected[col] = df_selected[col].ffill().bfill()

# 4. Добавляем обязательную колонку id
df_selected['id'] = 'gold'

# 5. Сортируем (это важно для Chronos)
df_selected = df_selected.sort_values('timestamp')

print(f"📈 Данные: {len(df_selected)} дней, с {df_selected['timestamp'].min()} по {df_selected['timestamp'].max()}")

# 6. Загружаем модель
print("🧠 Загружаем модель Chronos-2...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)

# 7. ИСПРАВЛЕННЫЙ ВЫЗОВ: Берем из официального примера
print("🔮 Делаем прогноз на 30 дней...")
pred_df = pipeline.predict_df(
    df_selected,
    prediction_length=30,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target=features[0],  # берем только 'Close' для первого раза
)

# 8. Визуализация
plt.figure(figsize=(12, 6))

# Исторические данные
plt.plot(df_selected['timestamp'], df_selected['Close'], label="Исторические данные", color='blue')

# Прогноз для золота
gold_forecast = pred_df[pred_df['id'] == 'gold']

plt.plot(gold_forecast['timestamp'], gold_forecast['0.5'], 
         label="Прогноз (медиана)", color='red', linewidth=2)

plt.fill_between(
    gold_forecast['timestamp'],
    gold_forecast['0.1'],
    gold_forecast['0.9'],
    color='red', alpha=0.2, label="Доверительный интервал (10%-90%)"
)

plt.title("Прогноз золота с Chronos-2 (финальная версия)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_chronos2_final.png')
print("✅ График сохранён: data/forecast_chronos2_final.png")
plt.show()