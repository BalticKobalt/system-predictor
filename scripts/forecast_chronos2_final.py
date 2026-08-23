import pandas as pd
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

print("📊 Загружаем данные...")

# 1. Загружаем данные с правильным парсингом дат
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Выбираем нужные колонки и сбрасываем индекс
# ВАЖНО: после reset_index() дата становится обычной колонкой 'index'
features = ['Close', 'brent', 'vix', 'dxy', 'ai_gpr']
df_selected = df[features].copy().reset_index()

# 3. Правильно называем колонку с датой
df_selected = df_selected.rename(columns={'index': 'timestamp'})

# 4. Заполняем пропуски (сначала ffill, потом bfill — как в примерах Chronos)
print("🧹 Заполняем пропуски...")
df_selected = df_selected.ffill().bfill()

# 5. Добавляем item_id (обязательно, даже для одного ряда) [citation:1][citation:3]
df_selected['item_id'] = 'gold'

# 6. Сортируем по дате
print("Колонки до проверки:", df_selected.columns.tolist())
# Если колонка называется 'Date', а не 'timestamp':
if 'Date' in df_selected.columns:
    df_selected = df_selected.rename(columns={'Date': 'timestamp'})
df_selected = df_selected.sort_values('timestamp').reset_index(drop=True)

print(f"📈 Данные: {len(df_selected)} дней, с {df_selected['timestamp'].min()} по {df_selected['timestamp'].max()}")
print(f"📋 Колонки: {df_selected.columns.tolist()}")

# 7. Загружаем модель
print("🧠 Загружаем модель Chronos-2...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)

# 8. Делаем прогноз — используем predict_df как в официальных примерах [citation:1][citation:4]
print("🔮 Делаем прогноз на 30 дней...")
pred_df = pipeline.predict_df(
    df_selected,
    prediction_length=30,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="item_id",          # Название колонки с ID
    timestamp_column="timestamp", # Название колонки с датой
    target="Close",               # Что прогнозируем
    cross_learning=False          # Для одного ряда отключаем
)

# 9. Визуализация
plt.figure(figsize=(12, 6))

gold_historical = df_selected[df_selected['item_id'] == 'gold']
plt.plot(gold_historical['timestamp'], gold_historical['Close'], label="Исторические данные", color='blue')

gold_forecast = pred_df[pred_df['item_id'] == 'gold']
plt.plot(gold_forecast['timestamp'], gold_forecast['0.5'], 
         label="Прогноз (медиана)", color='red', linewidth=2)

plt.fill_between(
    gold_forecast['timestamp'],
    gold_forecast['0.1'],
    gold_forecast['0.9'],
    color='red', alpha=0.2, label="Доверительный интервал (10%-90%)"
)

plt.title("Прогноз цены золота с Chronos-2 (с ковариатами)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_chronos2_final.png')
print("✅ График сохранён: data/forecast_chronos2_final.png")
plt.show()

print("\n📊 Прогноз цены золота на 30 дней (медиана):")
print(gold_forecast[['timestamp', '0.5']])