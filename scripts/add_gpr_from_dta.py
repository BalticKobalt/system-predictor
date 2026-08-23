import pandas as pd
import os

print("📊 Добавляем GPR из .dta файла...")

# 1. Загружаем master_data
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Загружаем GPR из .dta файла
dta_file = 'data/data_gpr_export.dta'

if os.path.exists(dta_file):
    print(f"✅ Найден файл: {dta_file}")
    
    # Читаем .dta файл через pandas
    gpr_df = pd.read_stata(dta_file)
    
    print(f"📋 Колонки в GPR: {gpr_df.columns.tolist()}")
    print(f"📊 Первые 5 строк:\n{gpr_df.head(5)}")
    
    # Ищем колонку с датой
    date_col = None
    for col in gpr_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col is None:
        date_col = gpr_df.columns[0]
        print(f"⚠️ Использую первую колонку как дату: {date_col}")
    
    # Преобразуем дату
    gpr_df[date_col] = pd.to_datetime(gpr_df[date_col])
    gpr_df = gpr_df.set_index(date_col)
    
    # Ищем колонку с GPR
    gpr_col = None
    for col in gpr_df.columns:
        if 'gpr' in col.lower():
            gpr_col = col
            break
    
    if gpr_col is None:
        num_cols = gpr_df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            gpr_col = num_cols[0]
            print(f"⚠️ Использую колонку {gpr_col} как GPR")
    
    if gpr_col is not None:
        df['gpr'] = gpr_df[gpr_col]
        df.to_csv('data/master_data.csv')
        print(f"✅ GPR добавлен в master_data.csv")
        print(f"   Колонка: {gpr_col}")
        print(f"   Период: с {gpr_df.index.min()} по {gpr_df.index.max()}")
        print(f"   Количество записей: {len(gpr_df)}")
    else:
        print("❌ Не найдена колонка с GPR")
else:
    print(f"❌ Файл не найден: {dta_file}")