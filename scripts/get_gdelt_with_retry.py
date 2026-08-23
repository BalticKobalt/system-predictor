import requests
import pandas as pd
import os
import time

os.makedirs('../data', exist_ok=True)

def fetch_gdelt_with_retry(url, headers, max_retries=5):
    """Делает запрос к GDELT с умной обработкой ошибок"""
    wait_time = 30
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            # Если ответ пустой или не JSON
            if not response.text.strip():
                print(f"⚠️ Сервер вернул пустой ответ. Попытка {attempt+1}/{max_retries}")
                time.sleep(wait_time)
                wait_time *= 2
                continue
                
            # Пытаемся разобрать JSON
            try:
                data = response.json()
                return data
            except requests.exceptions.JSONDecodeError:
                print(f"⚠️ Сервер вернул не JSON. Ответ: {response.text[:100]}")
                time.sleep(wait_time)
                wait_time *= 2
                continue
                
        except requests.exceptions.Timeout:
            print(f"⚠️ Таймаут. Попытка {attempt+1}/{max_retries}")
            time.sleep(wait_time)
            wait_time *= 2
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Ошибка соединения. Попытка {attempt+1}/{max_retries}")
            time.sleep(wait_time)
            wait_time *= 2
            
    print("❌ Не удалось получить данные после всех попыток.")
    return None

print("Начинаем скачивание статей из GDELT...")

# Параметры запроса
query = "war OR conflict OR crisis"
url = "https://api.gdeltproject.org/api/v2/doc/doc?query=war OR conflict OR crisis&mode=artlist&format=json&maxrecords=250&startdatetime=20250101000000&enddatetime=20251231235959"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

data = fetch_gdelt_with_retry(url, headers)

if data:
    articles_list = data.get('articles', [])
    if articles_list:
        df = pd.DataFrame(articles_list)
        df.to_csv('../data/gdelt_global_articles_2025.csv', index=False)
        print(f"✅ Готово! Скачано {len(df)} статей.")
        print("Файл сохранён: data/gdelt_global_articles_2025.csv")
    else:
        print("⚠️ Статей по вашему запросу не найдено.")
else:
    print("❌ Не удалось получить данные.")