import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

print("🚀 Запуск ежедневного обновления прогноза...")
today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"📅 {today}")

# --- КОНФИГУРАЦИЯ ---
DATA_DIR = 'data/'
MASTER_PATH = DATA_DIR + 'master_data.csv'
WEIGHTS_PATH = DATA_DIR + 'lstm_weights_9features.pth'
FORECAST_PATH = DATA_DIR + 'forecast_result.json'
AI_GPR_URL = "https://www.matteoiacoviello.com/ai_gpr_data_daily.csv"
SEQ_LEN = 60

# --- ПРИЗНАКИ (9 колонок) ---
features = ['gold', 'brent', 'vix', 'dxy', 'ai_gpr', 'gpr', 'gecon', 'crisis_ratio', 'conflict_intensity']

# --- МОДЕЛЬ LSTM ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

# --- 1. ЗАГРУЗКА МАСТЕР-ТАБЛИЦЫ ---
print("\n📊 Загружаем master_data.csv...")
if os.path.exists(MASTER_PATH):
    df = pd.read_csv(MASTER_PATH, index_col=0, parse_dates=True)
    print(f"✅ Загружено {len(df)} строк, с {df.index.min()} по {df.index.max()}")
else:
    print("❌ master_data.csv не найден! Создаю из исходников...")
    # Здесь можно добавить код для создания из базовых файлов
    exit()

# --- 2. ОБНОВЛЕНИЕ ЦЕН (ежедневно) ---
print("\n📈 Обновляем цены через yfinance...")
try:
    # Скачиваем последние данные
    gold_new = yf.download("GC=F", period="5d")['Close'].rename('gold')
    brent_new = yf.download("BZ=F", period="5d")['Close'].rename('brent')
    vix_new = yf.download("^VIX", period="5d")['Close'].rename('vix')
    dxy_new = yf.download("DX-Y.NYB", period="5d")['Close'].rename('dxy')
    
    # Объединяем новые данные
    new_prices = pd.concat([gold_new, brent_new, vix_new, dxy_new], axis=1).dropna()
    print(f"✅ Загружено {len(new_prices)} новых дней")
    
    # Добавляем в мастер-таблицу
    for col in ['gold', 'brent', 'vix', 'dxy']:
        if col in new_prices.columns:
            # Обновляем только новые даты
            for date in new_prices.index:
                if date not in df.index:
                    df.loc[date, col] = new_prices.loc[date, col]
    
    print("✅ Цены обновлены")
except Exception as e:
    print(f"⚠️ Ошибка обновления цен: {e}")

# --- 3. ОБНОВЛЕНИЕ AI-GPR (раз в месяц) ---
# Проверяем, когда последний раз обновлялся AI-GPR
last_ai_update = None
if 'ai_gpr_last_update' in df.attrs:
    last_ai_update = df.attrs['ai_gpr_last_update']
else:
    # Проверяем по дате последней записи в ai_gpr
    ai_dates = df['ai_gpr'].dropna()
    if len(ai_dates) > 0:
        last_ai_update = ai_dates.index[-1]

# Если прошло больше 30 дней или данных нет — скачиваем
need_ai_update = True
if last_ai_update is not None:
    days_since = (datetime.now() - last_ai_update).days
    if days_since < 30:
        need_ai_update = False
        print(f"⏳ AI-GPR обновлялся {days_since} дней назад (обновление раз в месяц)")

if need_ai_update:
    print("\n📰 Скачиваем свежий AI-GPR...")
    try:
        response = requests.get(AI_GPR_URL, timeout=60)
        if response.status_code == 200:
            # Сохраняем временный файл
            temp_path = DATA_DIR + 'ai_gpr_temp.csv'
            with open(temp_path, 'w') as f:
                f.write(response.text)
            
            # Загружаем
            ai_gpr_new = pd.read_csv(temp_path, parse_dates=['Date'], index_col='Date')
            ai_gpr_new = ai_gpr_new[['GPR_AI']].rename(columns={'GPR_AI': 'ai_gpr'})
            
            # Обновляем в мастер-таблице
            for date in ai_gpr_new.index:
                if date in df.index:
                    df.loc[date, 'ai_gpr'] = ai_gpr_new.loc[date, 'ai_gpr']
                else:
                    # Если даты нет в df, добавляем (но это редко)
                    df.loc[date, 'ai_gpr'] = ai_gpr_new.loc[date, 'ai_gpr']
            
            # Сохраняем дату обновления
            df.attrs['ai_gpr_last_update'] = datetime.now()
            
            # Удаляем временный файл
            os.remove(temp_path)
            print("✅ AI-GPR обновлён")
        else:
            print(f"⚠️ Не удалось скачать AI-GPR: статус {response.status_code}")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки AI-GPR: {e}")
else:
    print("⏳ AI-GPR актуален, пропускаем")

# --- 4. СОХРАНЯЕМ ОБНОВЛЁННУЮ ТАБЛИЦУ ---
# Заполняем пропуски
df = df.ffill().bfill().fillna(0)

# Сохраняем
df.to_csv(MASTER_PATH)
print(f"\n✅ master_data.csv сохранён: {len(df)} строк, с {df.index.min()} по {df.index.max()}")

# --- 5. ЗАГРУЗКА МОДЕЛИ И ПРОГНОЗ ---
print("\n🧠 Загружаем модель...")
df_selected = df[features].copy().dropna()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected.values)

model = LSTMPredictor(input_size=len(features))
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    print("✅ Веса модели загружены")
else:
    print("⚠️ Веса не найдены!")

model.eval()

# --- 6. ПРОГНОЗ ---
print("\n🔮 Делаем прогноз на 30 дней...")
last_seq = scaled_data[-SEQ_LEN:]
future_preds = []
current_seq = last_seq.copy()
for _ in range(30):
    with torch.no_grad():
        next_pred = model(torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0))
    future_preds.append(next_pred.item())
    new_row = current_seq[-1].copy()
    new_row[0] = next_pred.item()
    current_seq = np.vstack([current_seq[1:], new_row])

# Обратное масштабирование
dummy_future = np.zeros((len(future_preds), len(features)))
dummy_future[:, 0] = future_preds
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

# --- 7. РАСЧЁТ ИНДЕКСА ОПАСНОСТИ ---
today_price = future_prices[0]
week_price = future_prices[6]
change = (week_price - today_price) / today_price * 100

# Нормализуем в шкалу 1-100 (1 — хорошо, 100 — плохо)
# change от -10% до +10% → индекс от 100 до 1
if change >= 10:
    danger_index = 1
elif change <= -10:
    danger_index = 100
else:
    danger_index = 50 - (change / 10) * 49  # линейная интерполяция

danger_index = max(1, min(100, danger_index))  # Ограничиваем 1-100

# Определяем статус
if danger_index < 30:
    status = "🟢 Стабильно"
    color = "green"
elif danger_index < 60:
    status = "🟡 Напряжённо"
    color = "yellow"
else:
    status = "🔴 Критично"
    color = "red"

# --- 8. СОХРАНЯЕМ РЕЗУЛЬТАТ ---
result = {
    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "forecast": future_prices.tolist(),
    "danger_index": round(danger_index, 1),
    "danger_status": status,
    "danger_color": color,
    "change_percent": round(change, 2),
    "weekly_forecast": round(week_price, 2),
    "today_forecast": round(today_price, 2),
    "interpretation": f"{status}. Изменение за неделю: {round(change, 2)}%"
}

with open(FORECAST_PATH, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ Прогноз сохранён в {FORECAST_PATH}")
print(f"📊 Индекс опасности: {result['danger_index']} — {result['danger_status']}")

print("\n🚀 Скрипт завершён успешно!")