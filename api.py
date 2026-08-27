import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ===== ПЕРЕКЛЮЧАЕМ ВЫВОД НА UTF-8 (чтобы emoji не падали в Windows-консоли) =====
import sys, io
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = FastAPI(title="System Predictor API", version="0.1")

# ===== РАЗРЕШАЕМ CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== КОНФИГУРАЦИЯ =====
DATA_PATH = 'data/master_data.csv'
WEIGHTS_PATH = 'data/lstm_weights_9features.pth'
FORECAST_PATH = 'data/forecast_result.json'
SEQ_LEN = 60

# ===== ОПРЕДЕЛЕНИЕ МОДЕЛИ (если понадобится) =====
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

# ===== ЗАГРУЗКА МОДЕЛИ (если нужно) =====
model_loaded = False
model = None
scaler = None
df_selected = None
features = None

def load_model():
    global model_loaded, model, scaler, df_selected, features
    if model_loaded:
        return
    
    print("🚀 Загружаем модель (для резерва)...")
    try:
        if not os.path.exists(DATA_PATH):
            print("   ⚠️ master_data.csv не найден")
            return
        
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        features = ['gold', 'brent', 'vix', 'dxy', 'ai_gpr', 'gpr', 'gecon', 'crisis_ratio', 'conflict_intensity']
        df_selected = df[features].copy().ffill().bfill().fillna(0)
        
        scaler = MinMaxScaler()
        scaler.fit_transform(df_selected.values)
        
        model = LSTMPredictor(input_size=len(features))
        if os.path.exists(WEIGHTS_PATH):
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
            model.eval()
            print("   ✅ Веса загружены (резерв)")
        else:
            print("   ⚠️ Веса не найдены")
        
        model_loaded = True
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")

# ===== ЭНДПОИНТЫ =====
@app.get("/")
def root():
    return {
        "message": "System Predictor API is running",
        "version": "0.2",
        "mode": "file-based (forecast_result.json)"
    }

@app.get("/forecast")
def get_forecast():
    """Возвращает прогноз из forecast_result.json"""
    try:
        if os.path.exists(FORECAST_PATH):
            with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "source": "forecast_result.json",
                **data
            }
        else:
            # Резерв: если файла нет, пытаемся считать на лету
            load_model()
            if model_loaded and model is not None:
                # ... (старый код расчёта)
                return {"error": "Файл не найден, расчёт пока не реализован в этой версии"}
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": "forecast_result.json not found and model not loaded"}
                )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "forecast_exists": os.path.exists(FORECAST_PATH),
        "model_loaded": model_loaded
    }

@app.get("/features")
def get_features():
    return {
        "features": [
            "gold", "brent", "vix", "dxy", "ai_gpr",
            "gpr", "gecon", "crisis_ratio", "conflict_intensity"
        ]
    }

@app.post("/update")
def trigger_update():
    """Запускает пересчёт прогноза в фоне (используется внешним ежедневным cron-ом).
    Важно: пересчёт идёт в том же процессе (в отдельном потоке), чтобы не грузить
    torch второй раз — на бесплатном тарифе Render это привело бы к OOM."""
    def _run():
        try:
            from update_forecast import run_update
            run_update()
        except Exception as e:
            print(f"❌ Ошибка пересчёта: {e}")
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "note": "Пересчёт запущен в фоне, файл обновится через ~1-2 мин."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)