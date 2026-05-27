from src.metrics_basic import calculate_avg_sentence_length, calculate_sentence_length_variance
from src.metrics_advanced import calculate_ttr, calculate_adjective_density

# Текст людини (нерівні речення, емоційні прикметники)
text_human = (
    "Furthermore, reducing love strictly to romance does it a great disservice. "
    "Love is a vast spectrum. "
    "There is the fierce, protective love of a parent for a child; the fierce loyalty of lifelong friendship; "
    "and perhaps most importantly, self-love, which forms the foundation of how we allow ourselves to be treated by the world."
)

# Текст ШІ (ідеально збалансовані за довжиною речення)
text_ai = (
    "Advantages of Limiting Car Usage. "
    "Limiting car usage has become a growing trend in many parts of the world. "
    "The idea is being implemented in suburban areas as a part of the movement called smart planning. "
    "There are several advantages to limiting car usage, including reduced greenhouse gas emissions."
)

print("🟢 HUMAN TEXT LINGUISTIC PROFILE:")
print(f"   • Avg Sentence Length:     {calculate_avg_sentence_length(text_human)} words")
print(f"   • Sentence Length Variance: {calculate_sentence_length_variance(text_human)} (High = Human)")
print(f"   • Lexical Richness (TTR):  {calculate_ttr(text_human)}")
print(f"   • Adjective Density:       {calculate_adjective_density(text_human)}")

print("\n🤖 AI TEXT LINGUISTIC PROFILE:")
print(f"   • Avg Sentence Length:     {calculate_avg_sentence_length(text_ai)} words")
print(f"   • Sentence Length Variance: {calculate_sentence_length_variance(text_ai)} (Low = AI)")
print(f"   • Lexical Richness (TTR):  {calculate_ttr(text_ai)}")
print(f"   • Adjective Density:       {calculate_adjective_density(text_ai)}")