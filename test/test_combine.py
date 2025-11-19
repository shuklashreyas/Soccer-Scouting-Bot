from test_intent import model, vectorizer
from nlp.entity_extraction import extract_entities

query = "Compare Haaland and Mbappé"
intent = model.predict(vectorizer.transform([query]))[0]
entities = extract_entities(query)

print(f"Intent → {intent}")
print(f"Entities → {entities}")
