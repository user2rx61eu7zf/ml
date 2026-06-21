import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

# 1. Chargement et Nettoyage
df = pd.read_csv("jdid.csv")
df = df.dropna(subset=["Class", "Payloads"])
df["Payloads"] = df["Payloads"].str.replace("xss", "", case=False, regex=True)

# Nettoyage des classes orphelines
class_counts = df["Class"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df["Class"].isin(valid_classes)]


Payload_train, Payload_test, Class_train, Class_test = train_test_split(
    df.Payloads,
    df.Class,
    test_size=0.99, 
    stratify=df.Class,
    random_state=42
)

vectorizer = TfidfVectorizer(
    analyzer='word',     
    ngram_range=(2,5),   
    max_features=5000,     
    min_df=2,
    dtype=np.float32
)

tfidf_train = vectorizer.fit_transform(Payload_train)
tfidf_test = vectorizer.transform(Payload_test)

# Conversion en dense pour Naive Bayes (GaussianNB ne supporte pas le sparse)
tfidf_train_dense = tfidf_train.toarray()
tfidf_test_dense = tfidf_test.toarray()

# 4. Définition des 5 modèles
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf', class_weight="balanced")
}

# 5. Entraînement et Évaluation
print(f"Volume d'entraînement : {Payload_train.shape[0]} échantillons")
print(f"Volume de test : {Payload_test.shape[0]} échantillons\n")

results = {}

for name, clf in models.items():
    print(f"--- Évaluation de : {name} ---")
    
    # Choix des données (dense pour Bayes, sparse pour les autres)
    X_train = tfidf_train_dense if name == "Naive Bayes" else tfidf_train
    X_test = tfidf_test_dense if name == "Naive Bayes" else tfidf_test
    
    clf.fit(X_train, Class_train)
    y_pred = clf.predict(X_test)
    
    print(classification_report(Class_test, y_pred))
    results[name] = f1_score(Class_test, y_pred, pos_label='Malicious')

print("\n--- Synthèse des F1-Scores (Classe Malveillante) ---")
for name, score in results.items():
    print(f"{name}: {score:.4f}")