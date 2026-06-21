import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
import matplotlib.pyplot as plt
import os

# 1. Chargement des données
train = pd.read_csv("XSSTraining.csv")
test = pd.read_csv("XSSTesting.csv")

X_train = train.drop("Class", axis=1)
y_train = train["Class"]
X_test = test.drop("Class", axis=1)
y_test = test["Class"]

feature_names = X_train.columns


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



print("=== RÉSULTATS RÉGRESSION LOGISTIQUE ===")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_lr))

# --- Random Forest ---
print("=== RÉSULTATS RANDOM FOREST ===")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train) 
y_pred_rf = rf.predict(X_test)
print("\n", classification_report(y_test, y_pred_rf))

# --- K-Nearest Neighbors (k-NN) ---
print("=== RÉSULTATS k-NN ===")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_knn))

# --- Naive Bayes ---
print("=== RÉSULTATS NAIVE BAYES ===")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_nb))

# --- SVM ---
print("=== RÉSULTATS SVM ===")
svm = SVC(kernel='rbf', class_weight="balanced", random_state=42)
svm.fit(X_train_scaled, y_train) 
y_pred_svm = svm.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_svm))


# =====================================================================
# === 2. VALIDATION CROISÉE STRATIFIÉE, MATRICES & REPORTS CUMULÉS  ===
# =====================================================================

print("\n" + "="*50)
print("=== ÉVALUATION PAR STRATIFIED K-FOLD (K=5) ===")
print("="*50)

# Sécurité : création du dossier pour stocker les images
os.makedirs("Figs", exist_ok=True)

# Préparation des structures globales requises pour le K-Fold
X_global_scaled = np.vstack((X_train_scaled, X_test_scaled))
X_global_raw = pd.concat([X_train, X_test], axis=0)
y_global = pd.concat([y_train, y_test], axis=0)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionnaire associant chaque modèle à sa matrice de données adéquate (scalée ou brute)
models_to_cv = {
    "Régression Logistique": (LogisticRegression(max_iter=1000, class_weight="balanced"), X_global_scaled),
    "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), X_global_scaled),
    "Naive Bayes": (GaussianNB(), X_global_scaled),
    "SVM (RBF)": (SVC(kernel='rbf', class_weight="balanced", random_state=42), X_global_scaled),
    "Random Forest": (RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42), X_global_raw)
}

# Étape A : Calcul et affichage des métriques moyennes (Moyenne ± Écart-type)
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

for name, (model, X_data) in models_to_cv.items():
    print(f"\nCalcul des métriques CV pour : {name}...")
    cv_results = cross_validate(model, X_data, y_global, cv=cv_strategy, scoring=metrics, n_jobs=-1)
    
    print(f"--- Résultats CV Moyens ({name}) ---")
    print(f"Accuracy  : {cv_results['test_accuracy'].mean():.4f} (± {cv_results['test_accuracy'].std():.4f})")
    print(f"Précision : {cv_results['test_precision_macro'].mean():.4f} (± {cv_results['test_precision_macro'].std():.4f})")
    print(f"Rappel    : {cv_results['test_recall_macro'].mean():.4f} (± {cv_results['test_recall_macro'].std():.4f})")
    print(f"F1-Score  : {cv_results['test_f1_macro'].mean():.4f} (± {cv_results['test_f1_macro'].std():.4f})")


print("\n" + "="*50)
print("=== GÉNÉRATION DES MATRICES DE CONFUSION ET RAPPORTS CUMULÉS ===")
print("="*50)

# Étape B : Génération des prédictions cumulées, affichage des rapports et sauvegarde des figures
for name, (model, X_data) in models_to_cv.items():
    print(f"\n" + "-"*40)
    print(f"Calcul complet pour le modèle : {name}...")
    print("-"*40)
    
    # 1. Génération des prédictions globales par validation croisée
    y_pred_cv = cross_val_predict(model, X_data, y_global, cv=cv_strategy, n_jobs=-1)
    
    # 2. Calcul et affichage de la matrice de confusion cumulée
    cm_cv = confusion_matrix(y_global, y_pred_cv, labels=['Benign', 'Malicious'])
    print(f"\n[+] Matrice de confusion cumulée ({name}) :")
    print(cm_cv)
    
    # 3. Calcul et affichage du Classification Report par classe (Ajouté avec succès !)
    print(f"\n[+] Rapport de classification cumulé ({name}) :")
    print(classification_report(y_global, y_pred_cv, target_names=['Benign', 'Malicious'], digits=4))
    
    # 4. Génération et sauvegarde automatique de la figure pour LaTeX
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=['Benign', 'Malicious'])
    disp.plot(cmap='Greys', ax=ax)
    plt.title(f"Matrice de confusion K-Fold Cumulée\n{name}")
    
    # Nettoyage du nom pour le fichier image
    clean_name = name.lower().replace(' ', '').replace('(', '').replace(')', '').replace('é', 'e').replace('-', '')
    filename = f"Figs/mat-cv-{clean_name}.png"
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Figure sauvegardée sous : {filename}")