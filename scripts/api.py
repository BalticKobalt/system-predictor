import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

app = FastAPI(title="System Predictor API", version="0.1")

# --- Конфигурация ---
DATA_PATH = 'B:/PtFiles/data/master_data.csv'
WEIGHTS_PATH = 'B:/PtFiles/data/lstm_weights_9features.pth'
SEQ_LEN = 60

# --- Определение модели LSTM ---
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

# --- Загрузка данных и модели (при старте) ---
print("🚀 Загружаем API...")

# 1. Загружаем данные
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# 2. Признаки (9 колонок)
features = ['gold', 'brent', 'vix', 'dxy', 'ai_gpr', 'gpr', 'gecon', 'crisis_ratio', 'conflict_intensity']
df_selected = df[features].copy()

# 3. Заполняем пропуски
df_selected = df_selected.ffill().bfill().fillna(0)

print(f"📊 Загружено {len(df_selected)} строк, признаки: {df_selected.columns.tolist()}")

# 4. Нормализация
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected.values)

# 5. Создаём модель
input_size = len(features)
model = LSTMPredictor(input_size)

# 6. Загружаем веса
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    print("✅ Веса модели загружены")
else:
    print("⚠️ Веса не найдены, используется случайная инициализация")

model.eval()

# --- Функция для безопасного получения последней даты ---
def get_last_valid_date(df_selected):
    last_date = df_selected.index[-1]
    if isinstance(last_date, str):
        try:
            last_date = pd.to_datetime(last_date)
        except:
            last_date = datetime.now()
    elif pd.isna(last_date):
        last_date = datetime.now()
    return last_date

# --- Функция для прогноза ---
def make_forecast(days=30):
    last_seq = scaled_data[-SEQ_LEN:]
    future_preds = []
    current_seq = last_seq.copy()
    
    for _ in range(days):
        with torch.no_grad():
            next_pred = model(torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0))
        future_preds.append(next_pred.item())
        new_row = current_seq[-1].copy()
        new_row[0] = next_pred.item()
        current_seq = np.vstack([current_seq[1:], new_row])
    
    dummy_future = np.zeros((len(future_preds), len(features)))
    dummy_future[:, 0] = future_preds
    return scaler.inverse_transform(dummy_future)[:, 0]

# --- Функция интерпретации ---
def get_interpretation(prices, days):
    if len(prices) < 2:
        return "Недостаточно данных."
    change = (prices[-1] - prices[0]) / prices[0] * 100
    if change > 2:
        return f"📈 Ожидается рост цены золота на {change:.1f}% за {days} дней."
    elif change < -2:
        return f"📉 Ожидается падение цены золота на {abs(change):.1f}% за {days} дней."
    else:
        return f"⚖️ Цена золота останется стабильной (изменение {change:.1f}%) за {days} дней."

# --- Эндпоинты ---
@app.get("/")
def root():
    return {"message": "System Predictor API is running"}

@app.get("/forecast")
def get_forecast():
    try:
        forecast_30 = make_forecast(30)
        forecast_7 = forecast_30[:7]
        last_date = get_last_valid_date(df_selected)
        
        return {
            "last_update": last_date.strftime("%Y-%m-%d"),
            "weekly": {
                "gold": float(forecast_7[-1]),
                "days": 7,
                "interpretation": get_interpretation(forecast_7, 7)
            },
            "monthly": {
                "gold": float(forecast_30[-1]),
                "days": 30,
                "interpretation": get_interpretation(forecast_30, 30)
            },
            "full_forecast": forecast_30.tolist()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/chart")
def get_chart_url():
    return {"chart_url": "/static/forecast_chart.png"}

# --- Запуск ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)