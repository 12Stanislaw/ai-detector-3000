import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from src.classifier import run_full_model_test, train_detector, predict_ai_probability
from src.metrics_advanced import calculate_ttr, calculate_adjective_density

def analyze_text_action():
    """
    Реальна функція-обробник кнопки.
    Отримує текст, передає в ML-модель та оновлює GUI.
    """
    user_text = text_input.get("1.0", tk.END).strip()
    
    if not user_text:
        result_label.config(text="❌ Будь ласка, введіть текст для аналізу!", foreground="#ff5555")
        return
    
    # Викликаємо нашу модель і отримуємо реальний відсоток ШІ
    ai_percentage = predict_ai_probability(user_text)
    
    #Find ttr_score and adj_density from metrics advanced
    ttr_score = calculate_ttr(user_text)
    adj_density = calculate_adjective_density(user_text)

    # Твоя логіка світлофора, яку ти написав!
    if 65 < ai_percentage <= 100: 
        color = "#ff5555"      # Червоний
        verdict = "High Risk: AI Generated"
    elif 35 < ai_percentage <= 65:
        color = "#deff08"      # Жовтий
        verdict = "Suspicious: Mixed Content"
    else:
        color = "#05f234"      # Зелений
        verdict = "Safe: Human Written"

    # Виводимо результат на екран
    result_label.config(
        text=f"{verdict} ({ai_percentage}%)", 
        foreground=color
    )

    # Динамічно виводимо твої метрики у текстове поле картки результатів
    metrics_label.config(
        text=(
            f"📊 Text Style Metrics (Advanced Mode):\n"
            f"• Lexical Richness (TTR): {ttr_score:.4f}\n"
            f"• Adjective Density (POS): {adj_density * 100:.2f}%\n\n"
            f"💡 Info:\n"
            f"  - Lower TTR usually indicates repetitive AI patterns.\n"
            f"  - Human essays often have higher adjective density."
        )
    )
#--- ДИЗАЙН ВІКНА ---
root = tk.Tk()
root.title("AI-detector 3000")
root.geometry("700x550")
root.configure(bg="#1e1e2e") # Темно-синій/чорний фон (Сучасний вайб)

# Налаштування стилів для стандартних елементів
style = ttk.Style()
style.theme_use('default')
style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Arial", 11))
style.configure("TButton", background="#45475a", foreground="#ffffff", font=("Arial", 11, "bold"), borderwidth=0)
style.map("TButton", background=[("active", "#585b70")])

# 1. Заголовок програми
title_label = ttk.Label(
    root, 
    text="🔍 AI-detector 3000", 
    font=("Arial", 18, "bold"), 
    foreground="#cba6f7" # Фіолетовий акцент
)
title_label.pack(pady=15)

# 2. Підзаголовок (Слоган)
subtitle_label = ttk.Label(
    root, 
    text="Paste the text below to check if it has a human soul:",
    font=("Arial", 10, "italic")
)
subtitle_label.pack(pady=5)

# 3. Поле для вводу тексту (із прокруткою)
text_input = scrolledtext.ScrolledText(
    root, 
    wrap=tk.WORD, 
    width=75, 
    height=12, 
    font=("Arial", 11),
    bg="#313244", 
    fg="#cdd6f4", 
    insertbackground="white", # Колір курсору
    bd=0, 
    highlightthickness=1, 
    highlightbackground="#45475a"
)
text_input.pack(pady=15, padx=20)

# 4. Кнопка аналізу
analyze_button = ttk.Button(
    root, 
    text="Check text", 
    command=analyze_text_action, # При натисканні викличеться функція вище
    cursor="hand2"
)
analyze_button.pack(pady=10, ipady=5, ipadx=10)

# 5. Рамка для виводу результатів (Картка)
result_card = tk.Frame(root, bg="#313244", bd=0, highlightthickness=1, highlightbackground="#45475a")
result_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

# Головний вердикт всередині картки
result_label = ttk.Label(
    result_card, 
    text="Waiting for the text to be analyzed...", 
    font=("Arial", 13, "bold"), 
    background="#313244", 
    foreground="#a6adc8"
)
result_label.pack(pady=10, padx=10)

# Детальні метрики всередині картки
metrics_label = ttk.Label(
    result_card, 
    text="The metrics will appear after scanning", 
    font=("Arial", 10), 
    background="#313244", 
    foreground="#bac2de",
    justify=tk.LEFT
)
metrics_label.pack(pady=5, padx=10, anchor="w")

# Навчаємо модель безпосередньо перед запуском інтерфейсу
try:

    train_detector("dataset.csv")
    run_full_model_test("dataset.csv")
except Exception as e:
    print(f"Помилка ініціалізації моделі: {e}")

# Запуск головного циклу програми
root.mainloop()
