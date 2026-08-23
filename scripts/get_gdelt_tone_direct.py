import requests
import pandas as pd

print("📰 Скачиваем тональность из GDELT...")

url = "https://api.gdeltproject.org/api/v2/doc/doc"
params = {
    "query": "war OR conflict OR economy OR oil OR gold OR crisis",
    "mode": "timelinevol",
    "format": "json",
    "startdatetime": "2025-01-01",
    "enddatetime": "2026-08-21",
    "maxrecords": 1000
}

try:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    
    if 'timeline' in data and data['timeline']:
        df = pd.DataFrame(data['timeline'])
        df.to_csv('data/gdelt_tone_direct.csv', index=False)
        print(f"✅ Успех! Скачано {len(df)} дней.")
        print(df.head())
    else:
        print("⚠️ Данные не найдены.")
        print(data)
        
except requests.exceptions.RequestException as e:
    print(f"❌ Ошибка: {e}")