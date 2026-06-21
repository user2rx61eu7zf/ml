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


df = pd.read_csv("nv.csv")
df = df.dropna(subset=["Class", "Payloads"])
df["Payloads"] = df["Payloads"].str.replace("xss", "", case=False, regex=True)


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
    min_df=2,
    dtype=np.float32
)

tfidf_train = vectorizer.fit_transform(Payload_train)
tfidf_test = vectorizer.transform(Payload_test)


tfidf_train_dense = tfidf_train.toarray()
tfidf_test_dense = tfidf_test.toarray()


models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf', class_weight="balanced", random_state=42)
}


print(f"Volume d'entraînement : {Payload_train.shape[0]} échantillons")
print(f"Volume de test : {Payload_test.shape[0]} échantillons\n")

results = {}

for name, clf in models.items():
    print(f"--- Évaluation de : {name} ---")
    
   
    X_tr = tfidf_train_dense if name == "Naive Bayes" else tfidf_train
    X_te = tfidf_test_dense if name == "Naive Bayes" else tfidf_test
    
    clf.fit(X_tr, Class_train)
    y_pred = clf.predict(X_te)
    
    print(classification_report(Class_test, y_pred))
    results[name] = f1_score(Class_test, y_pred, pos_label='Malicious')

print("\n--- Synthèse des F1-Scores Classiques (Classe Malveillante) ---")
for name, score in results.items():
    print(f"{name}: {score:.4f}")



from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from scipy.sparse import vstack

print("\n" + "="*50)
print("=== ÉVALUATION TF-IDF PAR STRATIFIED K-FOLD (K=5) - TOUS LES ALGOS ===")
print("="*50)

os.makedirs("Figs_TFIDF", exist_ok=True)


X_tfidf_global_sparse = vstack([tfidf_train, tfidf_test])
X_tfidf_global_dense = X_tfidf_global_sparse.toarray()
y_tfidf_global = pd.concat([Class_train, Class_test], axis=0)


cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


models_tfidf_kfold = {
    "Logistic Regression (TF-IDF)": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest (TF-IDF)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "KNN (TF-IDF)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (TF-IDF)": GaussianNB(),
    "SVM (TF-IDF)": SVC(kernel='rbf', class_weight="balanced", random_state=42)
}

metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']


for name, model in models_tfidf_kfold.items():
    print(f"\n[+] Calcul de la validation croisée pour : {name}...")
    
   
    X_data = X_tfidf_global_dense if "Naive Bayes" in name else X_tfidf_global_sparse
        
    
    cv_results = cross_validate(model, X_data, y_tfidf_global, cv=cv_strategy, scoring=metrics, n_jobs=-1)
    
    print(f"--- Résultats CV ({name}) ---")
    print(f"Accuracy  : {cv_results['test_accuracy'].mean():.4f} (± {cv_results['test_accuracy'].std():.4f})")
    print(f"Précision : {cv_results['test_precision_macro'].mean():.4f} (± {cv_results['test_precision_macro'].std():.4f})")
    print(f"Rappel    : {cv_results['test_recall_macro'].mean():.4f} (± {cv_results['test_recall_macro'].std():.4f})")
    print(f"F1-Score  : {cv_results['test_f1_macro'].mean():.4f} (± {cv_results['test_f1_macro'].std():.4f})")

    print(f"[+] Génération de la matrice de confusion cumulée pour : {name}...")
    y_pred_cv = cross_val_predict(model, X_data, y_tfidf_global, cv=cv_strategy, n_jobs=-1)
    cm_cv = confusion_matrix(y_tfidf_global, y_pred_cv, labels=['Benign', 'Malicious'])
    
    print("Matrice de confusion brute calculée :")
    print(cm_cv)
    
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=['Benign', 'Malicious'])
    disp.plot(cmap='Greys', ax=ax)
    plt.title(f"Matrice de Confusion K-Fold (TF-IDF)\n{name}")
    
    
    clean_name = name.lower().replace(' ', '').replace('TL', '').replace('tl', '').replace('(', '').replace(')', '').replace('-', '')
    filename = f"Figs_TFIDF/mat-cv-{clean_name}.png"
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Figure sauvegardée avec succès sous : {filename}")   