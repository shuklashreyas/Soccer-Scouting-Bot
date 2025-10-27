import os
import pickle

# Resolve model paths relative to repository layout so the test works
# when invoked from any current working directory
here = os.path.dirname(__file__)
model_dir = os.path.abspath(os.path.join(here, "..", "src", "models"))
model_path = os.path.join(model_dir, "intent_model.pkl")
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Test examples
test_queries = [
    "Compare Haaland and Mbappé",
    "Show me Messi’s assists",
    "Find players like Pedri",
    "Would Mbappé fit in La Liga"
]

for query in test_queries:
    X = vectorizer.transform([query])
    intent = model.predict(X)[0]
    print(f"🗣️ {query} → 🎯 {intent}")
