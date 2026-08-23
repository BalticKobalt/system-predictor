import requests
import pandas as pd
import os
import time

os.makedirs('../data', exist_ok=True)

def fetch_gdelt_with_retry(url, headers, max_retries=3):
    """Делает запрос к GDELT, автоматически обрабатывая ошибку 429 (Too Many Requests)"""
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            # Сервер говорит, что мы слишком часто стучимся
            wait_time = 10  # ждём 10 секунд по умолчанию
            # Проверяем, не сказал ли сервер, сколько именно ждать
            if 'Retry-After' in response.headers:
                try:
                    wait_time = int(response.headers['Retry-After'])
                except ValueError:
                    wait_time = 10
            print(f"⚠️ Ошибка 429. Ждём {wait_time} секунд перед повторной попыткой...")
            time.sleep(wait_time)
        else:
            # Другие ошибки (404, 500 и т.д.)
            print(f"❌ Ошибка API: статус {response.status_code}")
            print("Ответ сервера:", response.text[:200])
            return None

    print("❌ Превышено количество попыток. Сервер всё ещё возвращает 429.")
    return None

print("Начинаем скачивание статей из GDELT с обработкой ошибки 429...")

# Параметры запроса
query = "war OR conflict OR crisis"
url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist&format=json&maxrecords=250&startdatetime=2025-01-01&enddatetime=2025-12-31"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = fetch_gdelt_with_retry(url, headers)

if response:
    data = response.json()
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