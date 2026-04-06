import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("nv.csv")
df = df.dropna(subset=["Class", "Payloads"]) 


Payload_train, Payload_test, Class_train, Class_test = train_test_split(
    df.Payloads,
    df.Class,
    test_size=0.2,
    stratify=df.Class, 
    random_state=42
)

print(f"Entraînement : {Payload_train.shape[0]} instances")
print(f"Test : {Payload_test.shape[0]} instances")


vectorizer = TfidfVectorizer(
    analyzer='char',      
    ngram_range=(3, 5),    
    max_features=5000
)


tfidf_train = vectorizer.fit_transform(Payload_train)
tfidf_test = vectorizer.transform(Payload_test)



model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(tfidf_train, Class_train)



y_pred = model.predict(tfidf_test)
print("\n--- Rapport de Classification ---")
print(classification_report(Class_test, y_pred))

