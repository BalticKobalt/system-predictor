import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os

print("📊 Загружаем данные...")

# 1. Загружаем данные
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Выбираем все доступные колонки
features = ['gold', 'brent', 'vix', 'dxy', 'ai_gpr', 'gpr', 'gecon', 'crisis_ratio', 'conflict_intensity']
df_selected = df[features].dropna()

print(f"📋 Используемые признаки: {features}")
print(f"📈 Данные: {len(df_selected)} дней, с {df_selected.index.min()} по {df_selected.index.max()}")

# 3. Нормализация
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected.values)

# 4. Создание последовательностей
SEQ_LEN = 60
def create_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # Прогнозируем gold (первая колонка)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

if len(X) == 0:
    print("❌ Недостаточно данных!")
    exit()

print(f"🧩 Создано {len(X)} последовательностей")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 5. Разделение на train/val/test
split_train = int(len(X) * 0.7)
split_val = int(len(X) * 0.85)
X_train = X[:split_train]
y_train = y[:split_train]
X_val = X[split_train:split_val]
y_val = y[split_train:split_val]
X_test = X[split_val:]
y_test = y[split_val:]

print(f"🧠 Обучающая выборка: {len(X_train)} последовательностей")
print(f"🧪 Валидационная выборка: {len(X_val)} последовательностей")
print(f"🧪 Тестовая выборка: {len(X_test)} последовательностей")

# 6. Модель LSTM
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

model = LSTMPredictor(input_size=X.shape[2])

# 7. Обучение с EarlyStopping
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🔮 Обучаем модель (200 эпох с EarlyStopping)...")

best_val_loss = float('inf')
patience = 20
patience_counter = 0
best_model_state = None

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val.unsqueeze(1))
    
    if epoch % 10 == 0:
        print(f"  Эпоха {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  ⏹️ Early Stopping на эпохе {epoch}")
            break

model.load_state_dict(best_model_state)

# 8. Прогноз на будущее (30 дней)
model.eval()
last_seq = scaled_data[-SEQ_LEN:]
future_preds = []
for _ in range(30):
    with torch.no_grad():
        next_pred = model(torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0))
    future_preds.append(next_pred.item())
    new_row = last_seq[-1].copy()
    new_row[0] = next_pred.item()
    last_seq = np.vstack([last_seq[1:], new_row])

# Обратное масштабирование
dummy_future = np.zeros((len(future_preds), len(features)))
dummy_future[:, 0] = future_preds
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

# 9. Визуализация
plt.figure(figsize=(14, 7))

# Безопасно получаем последнюю дату
last_date = df_selected.index[-1]
if pd.isna(last_date) or isinstance(last_date, str):
    valid_dates = df_selected.index[~pd.isna(df_selected.index)]
    last_date = valid_dates[-1] if len(valid_dates) > 0 else None

if last_date is None:
    print("❌ Нет валидных дат!")
    exit()

# Исправлено: используем 'gold' вместо 'Close'
plt.plot(df_selected.index, df_selected['gold'], label="Исторические данные", color='blue', alpha=0.7)

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
plt.plot(future_dates, future_prices, label="Прогноз LSTM (улучшенный)", color='red', linewidth=2)

plt.title("Прогноз цены золота с LSTM (9 признаков)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_lstm_9features.png')
torch.save(model.state_dict(), 'data/lstm_weights_9features.pth')
print("✅ Веса модели сохранены: data/lstm_weights_9features.pth")
plt.show()

print("\n📊 Прогноз цены золота на 30 дней (LSTM):")
for i, val in enumerate(future_prices):
    print(f"  День {i+1}: {val:.2f}")