import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("📊 Загружаем данные...")

# 1. Загружаем данные
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Выбираем все доступные колонки (золото + другие признаки)
# Удаляем колонки с пропусками и берём только те, где есть данные
features = ['Close', 'brent', 'vix', 'dxy', 'ai_gpr']
df_selected = df[features].dropna()

print(f"📈 Данные: {len(df_selected)} дней, с {df_selected.index.min()} по {df_selected.index.max()}")
print(f"📋 Используемые признаки: {features}")

# 3. Нормализация данных (масштабирование в диапазон [0, 1])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_selected.values)

# 4. Создание последовательностей для LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  # все признаки за 60 дней
        y.append(data[i, 0])  # прогнозируем только золото (Close)
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  # используем 60 дней истории для прогноза
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Разделяем на тренировочную и тестовую выборки
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"🧠 Обучающая выборка: {len(X_train)} последовательностей")
print(f"🧪 Тестовая выборка: {len(X_test)} последовательностей")

# 5. Строим модель LSTM
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("🧠 Модель LSTM создана")

# 6. Обучение
print("🔮 Обучаем модель...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 7. Прогноз на тестовых данных
y_pred_scaled = model.predict(X_test)

# 8. Обратное масштабирование для золота
# Создаём пустой массив для обратного масштабирования
dummy = np.zeros((len(y_pred_scaled), len(features)))
dummy[:, 0] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy)[:, 0]

dummy_test = np.zeros((len(y_test), len(features)))
dummy_test[:, 0] = y_test
y_true = scaler.inverse_transform(dummy_test)[:, 0]

# 9. Прогноз на будущее (30 дней)
last_sequence = scaled_data[-SEQ_LENGTH:]
future_predictions = []
for _ in range(30):
    # Прогноз следующего дня
    next_pred_scaled = model.predict(last_sequence.reshape(1, SEQ_LENGTH, len(features)), verbose=0)
    future_predictions.append(next_pred_scaled[0, 0])
    
    # Обновляем последовательность: добавляем новый прогноз, убираем первый день
    # Для простоты используем последний прогноз только для золота, остальные признаки — последние известные
    new_row = last_sequence[-1].copy()
    new_row[0] = next_pred_scaled[0, 0]  # обновляем золото
    last_sequence = np.vstack([last_sequence[1:], new_row])

# Обратное масштабирование будущих прогнозов
dummy_future = np.zeros((len(future_predictions), len(features)))
dummy_future[:, 0] = future_predictions
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

# 10. Визуализация
plt.figure(figsize=(14, 7))

# Исторические данные (золото)
plt.plot(df_selected.index, df_selected['Close'], label="Исторические данные", color='blue', alpha=0.7)

# Прогноз на тестовых данных (для проверки качества)
test_start_idx = split_idx + SEQ_LENGTH
test_dates = df_selected.index[test_start_idx:test_start_idx + len(y_test)]
plt.plot(test_dates, y_pred, label="Прогноз (тест)", color='green', linestyle='--', alpha=0.7)

# Прогноз на будущее
future_dates = pd.date_range(start=df_selected.index[-1] + pd.Timedelta(days=1), periods=30)
plt.plot(future_dates, future_prices, label="Прогноз (будущее)", color='red', linewidth=2)

plt.title("Прогноз цены золота с LSTM (многомерный)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/forecast_lstm.png')
print("✅ График сохранён: data/forecast_lstm.png")
plt.show()

print("\n📊 Прогноз цены золота на 30 дней (LSTM):")
for i, val in enumerate(future_prices):
    print(f"  День {i+1}: {val:.2f}")