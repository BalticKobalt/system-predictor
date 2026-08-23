import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("📊 Загружаем данные...")

# 1. Загружаем данные
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Добавляем GPR и GECON (если есть в файлах)
# Проверяем, есть ли колонки с GPR и GECON
gpr_files = ['data/data_gpr_export.xls', 'data/gpr_data.xls']
gpr_data = None
for file in gpr_files:
    try:
        gpr_df = pd.read_excel(file, engine='openpyxl')
        # Пробуем найти колонку с датой и GPR
        if 'date' in gpr_df.columns or 'Date' in gpr_df.columns:
            date_col = 'date' if 'date' in gpr_df.columns else 'Date'
            gpr_df[date_col] = pd.to_datetime(gpr_df[date_col])
            gpr_df = gpr_df.set_index(date_col)
            # Ищем колонку с GPR
            for col in gpr_df.columns:
                if 'gpr' in col.lower():
                    gpr_data = gpr_df[[col]]
                    gpr_data.columns = ['gpr']
                    print(f"✅ Загружен GPR из {file}")
                    break
            break
    except:
        continue

# Загружаем GECON
gecon_files = ['data/GECON_indicator.xlsx', 'data/gecon.xlsx']
gecon_data = None
for file in gecon_files:
    try:
        gecon_df = pd.read_excel(file, engine='openpyxl')
        if 'date' in gecon_df.columns or 'Date' in gecon_df.columns:
            date_col = 'date' if 'date' in gecon_df.columns else 'Date'
            gecon_df[date_col] = pd.to_datetime(gecon_df[date_col])
            gecon_df = gecon_df.set_index(date_col)
            # Ищем колонку с GECON
            for col in gecon_df.columns:
                if 'gecon' in col.lower() or 'ec' in col.lower():
                    gecon_data = gecon_df[[col]]
                    gecon_data.columns = ['gecon']
                    print(f"✅ Загружен GECON из {file}")
                    break
    except:
        continue

# 3. Формируем финальный датасет
features = ['Close', 'brent', 'vix', 'dxy', 'ai_gpr']
df_selected = df[features].copy()

# Добавляем GPR и GECON, если они загружены
if gpr_data is not None:
    df_selected = df_selected.join(gpr_data, how='left')
    print(f"✅ Добавлен GPR: {len(gpr_data)} записей")
if gecon_data is not None:
    df_selected = df_selected.join(gecon_data, how='left')
    print(f"✅ Добавлен GECON: {len(gecon_data)} записей")

# Обновляем список признаков
features = df_selected.columns.tolist()
print(f"📋 Используемые признаки: {features}")

# 4. Заполняем пропуски
df_selected = df_selected.ffill().bfill().fillna(0)

# 5. Очищаем индекс
df_selected = df_selected[~df_selected.index.astype(str).str.contains('Date', case=False)]
df_selected.index = pd.to_datetime(df_selected.index, errors='coerce')
df_selected = df_selected.dropna()

print(f"📈 Данные: {len(df_selected)} дней, с {df_selected.index.min()} по {df_selected.index.max()}")

# 6. Нормализация
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected.values)

# 7. Создание последовательностей
SEQ_LEN = 60
def create_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # Прогнозируем Close (золото)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

if len(X) == 0:
    print("❌ Недостаточно данных!")
    exit()

print(f"🧩 Создано {len(X)} последовательностей")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 8. Разделение на train/val/test
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

# 9. Модель LSTM (улучшенная)
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

# 10. Обучение с EarlyStopping
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🔮 Обучаем модель (200 эпох с EarlyStopping)...")

best_val_loss = float('inf')
patience = 20
patience_counter = 0
best_model_state = None

for epoch in range(200):
    # Train
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val.unsqueeze(1))
    
    if epoch % 10 == 0:
        print(f"  Эпоха {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  ⏹️ Early Stopping на эпохе {epoch}")
            break

# Загружаем лучшую модель
model.load_state_dict(best_model_state)

# 11. Прогноз на будущее (30 дней)
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

# 12. Визуализация
plt.figure(figsize=(14, 7))

# Безопасно получаем последнюю дату
last_date = df_selected.index[-1]
if pd.isna(last_date) or isinstance(last_date, str):
    valid_dates = df_selected.index[~pd.isna(df_selected.index)]
    last_date = valid_dates[-1] if len(valid_dates) > 0 else None

if last_date is None:
    print("❌ Нет валидных дат!")
    exit()

plt.plot(df_selected.index, df_selected['Close'], label="Исторические данные", color='blue', alpha=0.7)

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
plt.plot(future_dates, future_prices, label="Прогноз LSTM (улучшенный)", color='red', linewidth=2)

plt.title("Прогноз цены золота с LSTM (GPR + GECON + 200 эпох)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_lstm_advanced.png')
print("✅ График сохранён: data/forecast_lstm_advanced.png")
plt.show()

print("\n📊 Прогноз цены золота на 30 дней (LSTM):")
for i, val in enumerate(future_prices):
    print(f"  День {i+1}: {val:.2f}")