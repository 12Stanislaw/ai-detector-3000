import os
import pandas as pd

BIG_FILE_PATH = "AI_Human.csv"  
SMALL_FILE_PATH = "dataset.csv"

def extract_balanced_chunks():
    if not os.path.exists(BIG_FILE_PATH):
        print(f"❌ Помилка: Файлу '{BIG_FILE_PATH}' не знайдено!")
        return

    print("⏳ Запускаю потоковий аналіз файлу (імітація LINQ)...")
    
    # Списки для збереження знайдених рядків
    human_rows = []
    ai_rows = []
    
    # Читаємо файл маленькими шматочками по 5000 рядків, щоб не вантажити залізо
    chunk_size = 5000
    chunk_iterator = pd.read_csv(BIG_FILE_PATH, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunk_iterator):
        # Перевіряємо, чи є взагалі така колонка у цьому шматку
        if 'generated' not in chunk.columns:
            print("❌ Помилка: Колонки 'generated' не знайдено у файлі!")
            return
            
        # Наш "LINQ" фільтр всередині поточного шматка:
        # Шукаємо нулі (приводимо до float, бо у тебе 0.0)
        zeros = chunk[chunk['generated'] == 0.0]
        # Шукаємо одиниці (1.0)
        ones = chunk[chunk['generated'] == 1.0]
        
        # Додаємо знайдене в наші списки, якщо ще не набрали ліміт
        if len(human_rows) < 250:
            human_rows.extend(zeros.to_dict(orient='records'))
        if len(ai_rows) < 250:
            ai_rows.extend(ones.to_dict(orient='records'))
            
        # Якщо набрали по 250 штук і там, і там — зупиняємо конвеєр (Break)
        if len(human_rows) >= 250 and len(ai_rows) >= 250:
            print(f"🎯 Ціль досягнута на шматку №{i+1}! Потрібна кількість даних зібрана.")
            break
            
        print(f"📦 Оброблено рядків: {(i+1)*chunk_size}... Знайдено людей: {min(len(human_rows), 250)}/250, ШІ: {min(len(ai_rows), 250)}/250", end="\r")

    # Обрізаємо списки строго до 250, якщо раптом закинуло трохи більше
    df_human = pd.DataFrame(human_rows[:250])
    df_ai = pd.DataFrame(ai_rows[:250])
    
    if len(df_human) < 250 or len(df_ai) < 250:
        print(f"\n⚠️ Попередження: Не вдалося знайти повний баланс. Люди: {len(df_human)}, ШІ: {len(df_ai)}")
    
    # Зліплюємо докупи
    df_final = pd.concat([df_human, df_ai], ignore_index=True)
    
    # Перемішуємо рядки, щоб нулі та одиниці не йшли підряд
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Зберігаємо в маленький файл
    df_final.to_csv(SMALL_FILE_PATH, index=False)
    
    print(f"\n🎉 Перемога! Створено чистий '{SMALL_FILE_PATH}' на {len(df_final)} рядків.")
    print(f"📊 Фінальний розподіл у файлі:\n{df_final['generated'].value_counts()}")

if __name__ == "__main__":
    extract_balanced_chunks()
