import yfinance as yf
import pandas as pd

print("Скачиваем данные по золоту...")
gold = yf.download("GC=F", period="max")
gold.to_csv("gold_prices.csv")
print("Готово! Файл gold_prices.csv сохранён.")