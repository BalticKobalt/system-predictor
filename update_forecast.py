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
WEIGHTS_PARAMS_PATH = DATA_DIR + 'feature_weights.csv'
FORECAST_PATH = DATA_DIR + 'forecast_result.json'
AI_GPR_URL = "https://www.matteoiacoviello.com/ai_gpr_data_daily.csv"
SEQ_LEN = 60

# --- ПРИЗНАКИ ---
features = ['gold', 'brent', 'vix', 'dxy', 'ai_gpr', 'gpr', 'gecon', 'crisis_ratio', 'conflict_intensity']

# --- МОДЕЛЬ LSTM (для прогноза цен) ---
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

# --- МОДЕЛЬ РИСКА (для прогноза уровня опасности) ---
class RiskPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 1. ЗАГРУЗКА ДАННЫХ ---
print("\n📊 Загружаем master_data.csv...")
df = pd.read_csv(MASTER_PATH, index_col=0, parse_dates=True)

# --- 2. ЗАГРУЗКА ВЕСОВ ПАРАМЕТРОВ ---
print("\n📊 Загружаем веса параметров...")
if os.path.exists(WEIGHTS_PARAMS_PATH):
    weights_df = pd.read_csv(WEIGHTS_PARAMS_PATH)
    if '2' in weights_df.columns:
        weights = {row['Unnamed: 0']: row['2'] for _, row in weights_df.iterrows()}
    else:
        weights = {row['Unnamed: 0']: row['1'] for _, row in weights_df.iterrows()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    print("   Веса (нормализованные):")
    for f, w in weights.items():
        print(f"     {f}: {w:.4f}")
else:
    print("⚠️ Веса не найдены, использую равные")
    weights = {f: 1/len(features) for f in features}

# --- 3. ОБНОВЛЕНИЕ ЦЕН ---
print("\n📈 Обновляем цены через yfinance...")
try:
    tickers = {
        'gold': 'GC=F',
        'brent': 'BZ=F',
        'vix': '^VIX',
        'dxy': 'DX-Y.NYB'
    }
    new_data = {}
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, period="5d", progress=False)
            if data is not None and not data.empty:
                data.columns = data.columns.droplevel(1)
                if 'Close' in data.columns:
                    series = data['Close'].rename(name)
                    new_data[name] = series
                    print(f"   {name}: загружено {len(series)} дней")
                else:
                    print(f"   ⚠️ {name}: колонка 'Close' не найдена")
            else:
                print(f"   ⚠️ {name}: данные не получены")
        except Exception as e:
            print(f"   ⚠️ {name}: ошибка - {e}")
    if new_data:
        new_prices = pd.DataFrame(new_data)
        for col in new_prices.columns:
            if col in df.columns:
                for date in new_prices.index:
                    if date not in df.index:
                        df.loc[date, col] = new_prices.loc[date, col]
        print("✅ Цены обновлены")
    else:
        print("⚠️ Новых данных по ценам нет")
except Exception as e:
    print(f"⚠️ Ошибка обновления цен: {e}")

# --- 4. ОБНОВЛЕНИЕ AI-GPR (раз в месяц) ---
last_ai_update = None
ai_dates = df['ai_gpr'].dropna()
if len(ai_dates) > 0:
    last_ai_update = ai_dates.index[-1]

if last_ai_update is None or (datetime.now() - last_ai_update).days >= 30:
    print("\n📰 Скачиваем свежий AI-GPR...")
    try:
        response = requests.get(AI_GPR_URL, timeout=60)
        if response.status_code == 200:
            temp_path = DATA_DIR + 'ai_gpr_temp.csv'
            with open(temp_path, 'w') as f:
                f.write(response.text)
            ai_gpr_new = pd.read_csv(temp_path, parse_dates=['Date'], index_col='Date')
            ai_gpr_new = ai_gpr_new[['GPR_AI']].rename(columns={'GPR_AI': 'ai_gpr'})
            for date in ai_gpr_new.index:
                if date in df.index:
                    df.loc[date, 'ai_gpr'] = ai_gpr_new.loc[date, 'ai_gpr']
                else:
                    df.loc[date, 'ai_gpr'] = ai_gpr_new.loc[date, 'ai_gpr']
            os.remove(temp_path)
            print("✅ AI-GPR обновлён")
        else:
            print(f"⚠️ Не удалось скачать AI-GPR: статус {response.status_code}")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки AI-GPR: {e}")
else:
    print(f"⏳ AI-GPR актуален (обновлён {last_ai_update.strftime('%Y-%m-%d')})")

# --- 5. СОХРАНЯЕМ ОБНОВЛЁННУЮ ТАБЛИЦУ ---
df = df.ffill().bfill().fillna(0)
df.to_csv(MASTER_PATH)
print(f"\n✅ master_data.csv сохранён: {len(df)} строк")

# --- 6. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ ---
print("\n🧠 Подготавливаем данные для модели...")
df_selected = df[features].copy().dropna()

# Добавляем risk_level для дообучения
if 'risk_level' not in df_selected.columns:
    crises = pd.read_csv(DATA_DIR + 'crisis_events.csv', parse_dates=['start_date', 'end_date'])
    df['risk_level'] = 0
    for _, row in crises.iterrows():
        mask = (df.index >= row['start_date']) & (df.index <= row['end_date'])
        df.loc[mask, 'risk_level'] = row['level']
    df_selected['risk_level'] = df['risk_level']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected[features].values)

# --- 7. ЗАГРУЗКА LSTM ---
model = LSTMPredictor(input_size=len(features))
if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        print("✅ Веса LSTM загружены")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки весов LSTM: {e}")
else:
    print("⚠️ Веса LSTM не найдены")
model.eval()

# --- 8. ПРОГНОЗ НА 30 ДНЕЙ ---
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
dummy_future = np.zeros((len(future_preds), len(features)))
dummy_future[:, 0] = future_preds
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

# --- 9. РАСЧЁТ ИНДЕКСА (с дообучением) ---
def get_interpretation(prices, days):
    if len(prices) < 2:
        return "Недостаточно данных."
    change = (prices[-1] - prices[0]) / prices[0] * 100
    if change > 2:
        return f"📈 Ожидается рост на {change:.1f}% за {days} дней."
    elif change < -2:
        return f"📉 Ожидается падение на {abs(change):.1f}% за {days} дней."
    else:
        return f"⚖️ Стабильно (изменение {change:.1f}%) за {days} дней."

print("\n📊 Рассчитываем Индекс(ГН) с дообучением...")
risk_model_path = DATA_DIR + 'risk_model.pth'
risk_model_loaded = False

if os.path.exists(risk_model_path):
    try:
        # 1. Загружаем модель
        risk_checkpoint = torch.load(risk_model_path, map_location='cpu', weights_only=False)
        risk_model = RiskPredictor(input_size=len(features), num_classes=3)
        risk_model.load_state_dict(risk_checkpoint['model_state'])
        risk_model.train()
        
        # 2. Подготавливаем данные для дообучения
        train_seq_len = 60
        if len(scaled_data) >= train_seq_len + 1:
            X_train = []
            y_train = []
            
            for i in range(train_seq_len, len(scaled_data)):
                X_train.append(scaled_data[i-train_seq_len:i])
                # Используем реальные значения risk_level
                y_train.append(df_selected['risk_level'].iloc[i])
            
            if len(X_train) > 0:
                X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
                y_train = torch.tensor(np.array(y_train), dtype=torch.long)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(risk_model.parameters(), lr=0.0005)
                
                print("   🔄 Дообучаем модель на новых данных...")
                for epoch in range(5):
                    optimizer.zero_grad()
                    output = risk_model(X_train)
                    loss = criterion(output, y_train)
                    loss.backward()
                    optimizer.step()
                    print(f"      Эпоха {epoch+1}/5, Loss: {loss.item():.4f}")
                
                torch.save({
                    'model_state': risk_model.state_dict(),
                    'scaler': scaler,
                    'features': features,
                    'seq_len': train_seq_len
                }, risk_model_path)
                print("   ✅ Модель дообучена и сохранена")
        
        # 3. Делаем прогноз
        risk_model.eval()
        if len(scaled_data) >= train_seq_len:
            last_seq_risk = scaled_data[-train_seq_len:]
            with torch.no_grad():
                risk_input = torch.tensor(last_seq_risk, dtype=torch.float32).unsqueeze(0)
                risk_output = risk_model(risk_input)
                risk_probs = torch.softmax(risk_output, dim=1)
                
                danger_index = (risk_probs[0, 0] * 10 +
                              risk_probs[0, 1] * 50 +
                              risk_probs[0, 2] * 90)
                danger_index = danger_index.item()
                amplified_change = (danger_index - 50) / 50
                
                if danger_index > 65:
                    status = "🔴 Критично"
                elif danger_index > 35:
                    status = "🟡 Напряжённо"
                else:
                    status = "🟢 Спокойно"
                
                print(f"   Вероятности: Спокойно {risk_probs[0,0].item():.2f}, Напряжённо {risk_probs[0,1].item():.2f}, Критично {risk_probs[0,2].item():.2f}")
                print(f"   Индекс: {danger_index:.3f}")
                risk_model_loaded = True
        else:
            print("⚠️ Недостаточно данных для прогноза риска")
            
    except Exception as e:
        print(f"⚠️ Ошибка при дообучении/прогнозе: {e}")
        risk_model_loaded = False

# --- 10. ЗАПАСНОЙ ВАРИАНТ (если модель не загрузилась) ---
if not risk_model_loaded:
    print("⚠️ Использую формулу ×20 (запасной вариант)")
    
    if len(df_selected) >= 2:
        last_values = df_selected.iloc[-1]
        prev_values = df_selected.iloc[-2]
        
        changes = {}
        for feature in features:
            if prev_values[feature] != 0:
                changes[feature] = (last_values[feature] - prev_values[feature]) / abs(prev_values[feature])
            else:
                changes[feature] = 0
        
        base_change = sum(changes.get(f, 0) * weights.get(f, 0) for f in features)
        amplified_change = base_change * 20
        danger_index = 50 + amplified_change * 50
        danger_index = max(1, min(100, danger_index))
        danger_index = round(danger_index, 3)
        
        if danger_index > 65:
            status = "🔴 Критично"
        elif danger_index > 35:
            status = "🟡 Напряжённо"
        else:
            status = "🟢 Спокойно"
        
        print(f"   Базовое изменение: {base_change:.4f}")
        print(f"   Усиленное изменение (×20): {amplified_change:.4f}")
        print(f"   Индекс: {danger_index}")
    else:
        danger_index = 50.0
        status = "⚖️ Стабильно"
        amplified_change = 0
        changes = {}

# --- 11. СОХРАНЯЕМ РЕЗУЛЬТАТ ---
result = {
    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "danger_index": danger_index,
    "danger_status": status,
    "weighted_change": amplified_change,
    "changes": changes if 'changes' in locals() else {},
    "forecast": future_prices.tolist(),
    "weekly": {
        "gold": float(future_prices[6]) if len(future_prices) > 6 else None,
        "days": 7,
        "interpretation": get_interpretation(future_prices[:7], 7)
    },
    "monthly": {
        "gold": float(future_prices[29]) if len(future_prices) > 29 else None,
        "days": 30,
        "interpretation": get_interpretation(future_prices, 30)
    }
}

with open(FORECAST_PATH, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ Прогноз сохранён в {FORECAST_PATH}")
print(f"📊 Индекс(ГН): {danger_index} — {status}")
print(f"📈 Изменение: {amplified_change*100:.2f}%")
print("\n🚀 Скрипт завершён успешно!")