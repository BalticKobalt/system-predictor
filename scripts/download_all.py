import yfinance as yf
import pandas as pd
import os

# Создаём папку data, если её нет
os.makedirs("data", exist_ok=True)

# Список того, что качаем: (название, тикер, имя файла)
assets = [
    ("Золото", "GC=F", "gold_prices.csv"),
    ("Нефть Brent", "BZ=F", "brent_prices.csv"),
    ("Пшеница", "ZW=F", "wheat_prices.csv"),
    ("Индекс доллара", "DX-Y.NYB", "dxy_prices.csv"),
    ("Курс юаня", "USDCNY=X", "cny_prices.csv"),
    ("Курс рубля", "USDRUB=X", "rub_prices.csv"),
    ("VIX (индекс страха)", "^VIX", "vix_prices.csv"),
]

for name, ticker, filename in assets:
    print(f"Скачиваем {name} ({ticker})...")
    try:
        data = yf.download(ticker, period="max")
        data.to_csv(f"data/{filename}")
        print(f"✅ {name} сохранён в data/{filename}")
    except Exception as e:
        print(f"❌ Ошибка при скачивании {name}: {e}")

print("\n✅ Все доступные данные скачаны!")