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

# --- КОНФИГУРАЦИЯ ---
# Пути относительно корня проекта
DATA_PATH = 'data/master_data.csv'
WEIGHTS_PATH = 'data/lstm_weights_9features.pth'
SEQ_LEN = 60

# --- ОПРЕДЕЛЕНИЕ МОДЕЛИ LSTM ---
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

# --- ЗАГРУЗКА ДАННЫХ И МОДЕЛИ (при старте) ---
print("🚀 Загружаем API...")

# 1. Проверяем наличие файлов
if not os.path.exists(DATA_PATH):
    print(f"❌ Файл данных не найден: {DATA_PATH}")
    print("   Ожидаю данные в папке data/")
    
if not os.path.exists(WEIGHTS_PATH):
    print(f"⚠️ Веса модели не найдены: {WEIGHTS_PATH}")
    print("   Буду использовать случайную инициализацию")

# 2. Загружаем данные
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# 3. Признаки (9 колонок)
features = ['gold', 'brent', 'vix', 'dxy', 'ai_gpr', 'gpr', 'gecon', 'crisis_ratio', 'conflict_intensity']
df_selected = df[features].copy()

# 4. Заполняем пропуски
df_selected = df_selected.ffill().bfill().fillna(0)

print(f"📊 Загружено {len(df_selected)} строк, признаки: {df_selected.columns.tolist()}")

# 5. Нормализация
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected.values)

# 6. Создаём модель
input_size = len(features)
model = LSTMPredictor(input_size)

# 7. Загружаем веса (если есть)
if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        print("✅ Веса модели загружены")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки весов: {e}")
        print("   Использую случайную инициализацию")
else:
    print("⚠️ Веса не найдены, используется случайная инициализация")

model.eval()

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_last_valid_date(df_selected):
    """Безопасно возвращает последнюю валидную дату"""
    last_date = df_selected.index[-1]
    if isinstance(last_date, str):
        try:
            last_date = pd.to_datetime(last_date)
        except:
            last_date = datetime.now()
    elif pd.isna(last_date):
        last_date = datetime.now()
    return last_date

def make_forecast(days=30):
    """Делает прогноз на days дней вперёд"""
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

def get_interpretation(prices, days):
    """Генерирует текстовую интерпретацию прогноза"""
    if len(prices) < 2:
        return "Недостаточно данных."
    change = (prices[-1] - prices[0]) / prices[0] * 100
    if change > 2:
        return f"📈 Ожидается рост цены золота на {change:.1f}% за {days} дней."
    elif change < -2:
        return f"📉 Ожидается падение цены золота на {abs(change):.1f}% за {days} дней."
    else:
        return f"⚖️ Цена золота останется стабильной (изменение {change:.1f}%) за {days} дней."

# --- ЭНДПОИНТЫ API ---

@app.get("/")
def root():
    return {
        "message": "System Predictor API is running",
        "status": "online",
        "version": "0.1",
        "features": features
    }

@app.get("/forecast")
def get_forecast():
    """Возвращает прогноз на 7 и 30 дней"""
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

@app.get("/health")
def health_check():
    """Проверка работоспособности"""
    return {
        "status": "healthy",
        "data_rows": len(df_selected),
        "features_count": len(features),
        "weights_loaded": os.path.exists(WEIGHTS_PATH)
    }

@app.get("/features")
def get_features():
    """Возвращает список используемых признаков"""
    return {"features": features}

# --- ЗАПУСК ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)