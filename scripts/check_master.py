import pandas as pd

print("🔍 Проверяем master_data.csv...")

# Читаем без обработки дат
df = pd.read_csv('data/master_data.csv')

print(f"📊 Размер: {df.shape}")
print(f"📋 Колонки: {df.columns.tolist()}")
print(f"📋 Первые 5 строк:\n{df.head()}")
print(f"📋 Последние 5 строк:\n{df.tail()}")

# Проверяем колонку Date
if 'Date' in df.columns:
    print(f"\n📅 Тип колонки Date: {df['Date'].dtype}")
    print(f"📅 Примеры дат: {df['Date'].head(10).tolist()}")
    
    # Пробуем преобразовать
    try:
        dates = pd.to_datetime(df['Date'], errors='coerce')
        print(f"📅 Успешно преобразовано {dates.notna().sum()} из {len(dates)} дат")
        print(f"📅 Примеры: {dates.head(10).tolist()}")
    except Exception as e:
        print(f"❌ Ошибка преобразования: {e}")
else:
    print("❌ Колонка 'Date' не найдена!")