import os
import pandas as pd

# Назви твоїх файлів
old_file = "dataset.csv"  # Твій старий датасет, де точно є люди про машини
kaggle_file = "daigt_magic_generations.csv"  # Новий файл, де є ШІ на різні теми
output_file = "dataset_fused.csv"  # Фінальний зліпок

print("⏳ Крок 1: Дістаємо людей зі старого датасету...")
if not os.path.exists(old_file):
    print(f"❌ Помилка: Не знайдено старий файл {old_file}!")
    exit()

df_old = pd.read_csv(old_file)
# Перевіримо, яка там шапка. Якщо generated немає, спробуємо знайти потрібну колонку
if 'generated' not in df_old.columns and 'label' in df_old.columns:
    df_old = df_old.rename(columns={'label': 'generated'})

# Витягуємо тільки людей (де generated == 0)
df_human_pool = df_old[df_old['generated'] == 0]
print(f"   Знайдено людей у старому файлі: {len(df_human_pool)}")

print("\n⏳ Крок 2: Дістаємо ШІ з нового магічного датасету...")
df_new = pd.read_csv(kaggle_file)
if 'label' in df_new.columns:
    df_new = df_new.rename(columns={'label': 'generated'})

df_ai_pool = df_new[df_new['generated'] == 1]
print(f"   Знайдено ШІ в магічному файлі: {len(df_ai_pool)}")

# Визначаємо ліміт. Наприклад, візьмемо по 1000 або скільки максимально дозволяє кількість людей
limit = min(len(df_human_pool), len(df_ai_pool), 1000)

if limit == 0:
    print("❌ Помилка: Не вдалося зібрати пару. Один з пулів порожній.")
else:
    print(f"\n⚙️ Формуємо ідеальний баланс: беремо по {limit} зразків...")
    
    # Висмикуємо випадкові рядки для балансу
    df_human_sampled = df_human_pool.sample(n=limit, random_state=42)[['text', 'generated']]
    df_ai_sampled = df_ai_pool.sample(n=limit, random_state=42)[['text', 'generated']]
    
    # Зшиваємо їх разом
    df_final = pd.concat([df_human_sampled, df_ai_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Зберігаємо в ОРИГІНАЛЬНИЙ файл dataset.csv (щоб classifier.py його підхопив)
    df_final.to_csv("dataset.csv", index=False, encoding='utf-8')
    
    print(f"\n✅ Гібридний датасет успішно створено!")
    print(f"   📊 Загальна кількість рядків у dataset.csv: {len(df_final)} (50% Люди / 50% ШІ)")