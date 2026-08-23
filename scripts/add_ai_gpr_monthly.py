import pandas as pd

master = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

ai_monthly = pd.read_csv('data/ai_gpr_data_monthly.csv')
ai_monthly['Date'] = pd.to_datetime(ai_monthly['Date'])
ai_monthly = ai_monthly.set_index('Date')

# Берём колонку GPR_AI (или другую)
master['gpr_ai_monthly'] = ai_monthly['GPR_AI']

master.to_csv('data/master_data.csv')
print(f"✅ Добавлен месячный AI-GPR")
print(f"Колонки: {master.columns.tolist()}")