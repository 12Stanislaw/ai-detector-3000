from src.metrics_advanced import calculate_ttr, calculate_adjective_density

# Твій живий текст про кохання
text_human = "Love is the very essence of the human life. Without love, the world would become cold and bleak. God has gifted us different kinds of emotions and love is one the most beautiful of them all. It is an emotion that each of us has experienced at some point in our lives. When someone shows us their love, it makes us feel complete and special. It is like a divine energy that nourishes us throughout our lives. Love has a lot of positive aspects. It provides a foundation on which an individual builds, relishes, and nurtures. Furthermore, this intense feeling shows us how to deepen our emotions. We can say that giving love is a way of worshipping God."

# Шматок ШІ-тексту з нашого датасету
text_ai = """Love is perhaps the only human experience that is simultaneously a universal cliche and a deeply profound mystery. We see it plastered across billboards, sung about in pop songs, and written into the plots of every sitcom. Yet, when it hits you personally, it feels entirely brand new—as if you are the very first person in human history to discover it.

At its core, love is the ultimate defiance of human selfishness. We are wired for survival, programmed to look out for number one. But love flips that script. Whether it is the quiet, fierce devotion of a parent, the fierce loyalty of a lifelong friend, or the electric, terrifying vulnerability of romantic passion, love forces us to expand our definition of "self" to include someone else. Their pain becomes our pain; their joy becomes our joy.

However, we often do love a disservice by conflating it with romance. Romance is the spark—it is easy, chemical, and intoxicating. Love, on the other hand, is the fire that you have to keep feeding when the wood is damp and the wind is blowing. It is a choice disguised as an emotion. It means showing up when it is inconvenient, listening when you are tired, and choosing forgiveness over the hollow satisfaction of being right."""

print("🟢 HUMAN TEXT ANALYSIS:")
print(f"   Lexical Richness (TTR): {calculate_ttr(text_human)}")
print(f"   Adjective Density:      {calculate_adjective_density(text_human)}")

print("\n🤖 AI TEXT ANALYSIS:")
print(f"   Lexical Richness (TTR): {calculate_ttr(text_ai)}")
print(f"   Adjective Density:      {calculate_adjective_density(text_ai)}")
