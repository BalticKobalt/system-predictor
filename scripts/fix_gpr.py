import pandas as pd
import os

print("🔍 Пробуем прочитать GPR разными способами...")

# Путь к файлу
gpr_file = 'data/data_gpr_export.xls'

# Способ 1: как HTML
try:
    gpr = pd.read_html(gpr_file)
    if gpr:
        gpr = gpr[0]
        print(f"✅ Прочитан как HTML: {len(gpr)} строк")
        print(f"   Колонки: {gpr.columns.tolist()}")
        print(f"   Первые 3 строки:\n{gpr.head(3)}")
        
        # Пробуем найти дату и GPR
        date_col = None
        for col in gpr.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col is not None:
            gpr[date_col] = pd.to_datetime(gpr[date_col])
            gpr = gpr.set_index(date_col)
            
            # Ищем GPR
            gpr_col = None
            for col in gpr.columns:
                if 'gpr' in col.lower():
                    gpr_col = col
                    break
            
            if gpr_col is not None:
                # Добавляем в master_data
                master = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)
                master['gpr'] = gpr[gpr_col]
                master.to_csv('data/master_data.csv')
                print(f"✅ Добавлен GPR: {len(gpr)} записей")
                print(f"   Колонки в master_data: {master.columns.tolist()}")
            else:
                print("⚠️ Колонка GPR не найдена")
        else:
            print("⚠️ Колонка с датой не найдена")
            
except Exception as e:
    print(f"❌ Не удалось прочитать как HTML: {e}")

# Способ 2: с параметрами для старых Excel
try:
    gpr = pd.read_excel(gpr_file, engine='openpyxl', header=0)
    print(f"✅ Прочитан как Excel: {len(gpr)} строк")
    print(f"   Колонки: {gpr.columns.tolist()}")
except Exception as e:
    print(f"❌ Не удалось прочитать как Excel: {e}")

print("\n📊 Проверь файл вручную: открой data_gpr_export.xls в Excel и посмотри, что там.")