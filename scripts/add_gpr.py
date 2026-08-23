import pandas as pd
import os

print("📊 Добавляем GPR в master_data.csv...")

# 1. Загружаем master_data
df = pd.read_csv('data/master_data.csv', index_col=0, parse_dates=True)

# 2. Загружаем GPR
gpr_file = 'data/data_gpr_export.xls'

if os.path.exists(gpr_file):
    print(f"✅ Найден файл: {gpr_file}")
    
    # Пробуем прочитать как HTML (часто .xls от GPR — это HTML-таблица)
    try:
        print("🔍 Пробуем прочитать как HTML...")
        gpr_dfs = pd.read_html(gpr_file)
        if len(gpr_dfs) > 0:
            gpr_df = gpr_dfs[0]
            print(f"✅ Прочитан как HTML: {len(gpr_df)} строк")
        else:
            raise ValueError("Нет таблиц в HTML")
    except:
        # Если HTML не работает, пробуем как Excel с другим движком
        print("🔍 Пробуем прочитать как Excel с xlrd...")
        try:
            gpr_df = pd.read_excel(gpr_file, engine='xlrd')
        except:
            print("🔍 Пробуем прочитать как CSV...")
            try:
                gpr_df = pd.read_csv(gpr_file, sep='\t')
            except:
                print("❌ Не удалось прочитать файл")
                exit()
    
    print(f"📋 Колонки в GPR: {gpr_df.columns.tolist()}")
    print(f"📊 Первые 5 строк:\n{gpr_df.head(5)}")
    
    # Ищем колонку с датой
    date_col = None
    for col in gpr_df.columns:
        if 'date' in str(col).lower() or 'time' in str(col).lower():
            date_col = col
            break
    
    if date_col is None:
        # Пробуем первую колонку
        date_col = gpr_df.columns[0]
        print(f"⚠️ Использую первую колонку как дату: {date_col}")
    
    # Преобразуем дату
    try:
        gpr_df[date_col] = pd.to_datetime(gpr_df[date_col])
    except:
        # Если не получается, пробуем с форматом
        gpr_df[date_col] = pd.to_datetime(gpr_df[date_col], errors='coerce')
    
    gpr_df = gpr_df.set_index(date_col)
    
    # Удаляем строки с NaT
    gpr_df = gpr_df.dropna()
    
    # Ищем колонку с GPR
    gpr_col = None
    for col in gpr_df.columns:
        if 'gpr' in str(col).lower():
            gpr_col = col
            break
    
    if gpr_col is None:
        num_cols = gpr_df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            gpr_col = num_cols[0]
            print(f"⚠️ Использую колонку {gpr_col} как GPR")
    
    if gpr_col is not None:
        # Добавляем в master_data
        # Проверяем, есть ли уже колонка gpr
        if 'gpr' in df.columns:
            print("⚠️ Колонка gpr уже существует, перезаписываю...")
        
        df['gpr'] = gpr_df[gpr_col]
        df.to_csv('data/master_data.csv')
        print(f"✅ GPR добавлен в master_data.csv")
        print(f"   Колонка: {gpr_col}")
        print(f"   Период: с {gpr_df.index.min()} по {gpr_df.index.max()}")
        print(f"   Количество записей: {len(gpr_df)}")
    else:
        print("❌ Не найдена колонка с GPR")
else:
    print(f"❌ Файл не найден: {gpr_file}")