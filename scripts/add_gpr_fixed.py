import pandas as pd
import os

print("🔍 Поиск файлов в папке data...")

# Показываем все файлы в папке data
files = os.listdir('data')
print("Файлы в папке data:")
for f in files:
    print(f"  - {f}")

# Загружаем master_data
master = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 1. Находим файл GPR
gpr_files = [f for f in files if 'gpr' in f.lower() and (f.endswith('.xls') or f.endswith('.xlsx'))]
if gpr_files:
    gpr_file = gpr_files[0]
    print(f"\n✅ Найден файл GPR: {gpr_file}")
    
    try:
        gpr = pd.read_excel(f'data/{gpr_file}', engine='openpyxl')
        print(f"   Колонки в GPR: {gpr.columns.tolist()}")
        print(f"   Первые 3 строки:\n{gpr.head(3)}")
        
        # Ищем колонку с датой
        date_col = None
        for col in gpr.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            date_col = gpr.columns[0]
            print(f"⚠️ Использую первую колонку как дату: {date_col}")
        
        gpr[date_col] = pd.to_datetime(gpr[date_col])
        gpr = gpr.set_index(date_col)
        
        # Ищем колонку с GPR
        gpr_col = None
        for col in gpr.columns:
            if 'gpr' in col.lower() or 'risk' in col.lower():
                gpr_col = col
                break
        
        if gpr_col is None:
            num_cols = gpr.select_dtypes(include=['float64', 'int64']).columns
            if len(num_cols) > 0:
                gpr_col = num_cols[0]
                print(f"⚠️ Использую колонку {gpr_col} как GPR")
        
        if gpr_col is not None:
            master['gpr'] = gpr[gpr_col]
            print(f"✅ Добавлен GPR: {len(gpr)} записей")
        else:
            print("❌ Не найдена колонка с GPR")
            
    except Exception as e:
        print(f"❌ Ошибка при чтении GPR: {e}")
else:
    print("❌ Файл GPR не найден в папке data")

# 2. Находим файл AI-GPR
ai_files = [f for f in files if 'ai' in f.lower() and 'gpr' in f.lower() and f.endswith('.csv')]
if ai_files:
    ai_file = ai_files[0]
    print(f"\n✅ Найден файл AI-GPR: {ai_file}")
    
    try:
        ai_gpr = pd.read_csv(f'data/{ai_file}')
        print(f"   Колонки в AI-GPR: {ai_gpr.columns.tolist()}")
        print(f"   Первые 3 строки:\n{ai_gpr.head(3)}")
        
        date_col = None
        for col in ai_gpr.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            date_col = ai_gpr.columns[0]
            print(f"⚠️ Использую первую колонку как дату: {date_col}")
        
        ai_gpr[date_col] = pd.to_datetime(ai_gpr[date_col])
        ai_gpr = ai_gpr.set_index(date_col)
        
        ai_col = None
        for col in ai_gpr.columns:
            if 'ai' in col.lower() or 'gpt' in col.lower() or 'gpr' in col.lower():
                ai_col = col
                break
        
        if ai_col is None:
            num_cols = ai_gpr.select_dtypes(include=['float64', 'int64']).columns
            if len(num_cols) > 0:
                ai_col = num_cols[0]
                print(f"⚠️ Использую колонку {ai_col} как AI-GPR")
        
        if ai_col is not None:
            master['ai_gpr'] = ai_gpr[ai_col]
            print(f"✅ Добавлен AI-GPR: {len(ai_gpr)} записей")
        else:
            print("❌ Не найдена колонка с AI-GPR")
            
    except Exception as e:
        print(f"❌ Ошибка при чтении AI-GPR: {e}")
else:
    print("❌ Файл AI-GPR не найден в папке data")

# Сохраняем обновлённую таблицу
master.to_csv('data/master_data.csv')
print(f"\n✅ Таблица сохранена! Размер: {master.shape}")
print(f"Колонки: {master.columns.tolist()}")
print(f"Период: с {master.index.min()} по {master.index.max()}")