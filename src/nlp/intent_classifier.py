import pickle

with open("data/models/intent_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/models/intent_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

def predict_intent(query):
    X = vectorizer.transform([query])
    return clf.predict(X)[0]
