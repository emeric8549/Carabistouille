import os
import wikipediaapi

os.makedirs("data", exist_ok=True)

wiki = wikipediaapi.Wikipedia(
    language="fr",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="WikipediaPhysicsRAG"
)

topics = ["Physique", "Mécanique quantique", "Relativité restreinte", "Relativité générale", "Thermodynamique"]

corpus = []
for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        corpus.append(page.text)

with open("data/physics_articles.txt", "w+", encoding="utf-8") as f:
    for article in corpus:
        f.write(article + "\n\n")