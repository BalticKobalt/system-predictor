import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

print("🧠 Обучаем нейросетевую модель риска...")

# --- 1. ЗАГРУЗКА ДАННЫХ ---
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)
crises = pd.read_csv('data/crisis_events.csv', parse_dates=['start_date', 'end_date'])

# --- 2. СОЗДАЁМ ЦЕЛЕВУЮ ПЕРЕМЕННУЮ (уровень кризиса) ---
df['risk_level'] = 0
for _, row in crises.iterrows():
    mask = (df.index >= row['start_date']) & (df.index <= row['end_date'])
    df.loc[mask, 'risk_level'] = row['level']

# --- 3. ВЫБИРАЕМ ПРИЗНАКИ ---
features = ['gold', 'brent', 'vix', 'dxy', 'ai_gpr', 'gpr', 'gecon', 'crisis_ratio', 'conflict_intensity']
df_selected = df[features].copy()
df_selected['risk_level'] = df['risk_level']

# Удаляем строки с NaN
df_selected = df_selected.dropna()

print(f"📊 Данные: {len(df_selected)} строк, с {df_selected.index.min()} по {df_selected.index.max()}")

# --- 4. НОРМАЛИЗАЦИЯ ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected[features].values)

# --- 5. СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ---
SEQ_LEN = 60
def create_sequences(data, labels, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(labels[i])  # risk_level
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, df_selected['risk_level'].values)

print(f"🧩 Создано {len(X)} последовательностей")

# --- 6. РАЗДЕЛЕНИЕ НА ОБУЧЕНИЕ И ТЕСТ ---
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"🧠 Обучающая выборка: {len(X_train)}")
print(f"🧪 Тестовая выборка: {len(X_test)}")

# --- 7. ОПРЕДЕЛЕНИЕ МОДЕЛИ ---
class RiskPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = RiskPredictor(input_size=len(features), num_classes=3)

# --- 8. ОБУЧЕНИЕ ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🔮 Обучаем модель...")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = criterion(test_output, y_test)
            preds = torch.argmax(test_output, dim=1)
            acc = (preds == y_test).float().mean()
        print(f"  Эпоха {epoch}: Loss = {loss.item():.4f}, Test Acc = {acc.item():.4f}")

# --- 9. СОХРАНЕНИЕ МОДЕЛИ ---
torch.save({
    'model_state': model.state_dict(),
    'scaler': scaler,
    'features': features,
    'seq_len': SEQ_LEN
}, 'data/risk_model.pth')

print("\n✅ Модель сохранена в data/risk_model.pth")