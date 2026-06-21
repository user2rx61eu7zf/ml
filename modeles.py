import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.svm import SVC

train = pd.read_csv("XSSTraining.csv")
test = pd.read_csv("XSSTesting.csv")


X_train = train.drop("Class", axis=1)
y_train = train["Class"]
X_test = test.drop("Class", axis=1)
y_test = test["Class"]

feature_names = X_train.columns

#normalisation 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#regrssion logistique 
print("=== RÉSULTATS RÉGRESSION LOGISTIQUE ===")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_lr))
print("=== RÉSULTATS DÉTAILLÉS REGRESSION LOGISTIQUE (BRUTS) ===")
precision = precision_score(y_test, y_pred_lr, pos_label='Malicious')
recall = recall_score(y_test, y_pred_lr, pos_label='Malicious')
f1 = f1_score(y_test, y_pred_lr, pos_label='Malicious')
acc = accuracy_score(y_test, y_pred_lr)

print(f"Précision brute : {precision}")
print(f"Rappel brut    : {recall}")
print(f"F1-Score brut  : {f1}")
print(f"Accuracy brute  : {acc}")
"""
cm_lr = confusion_matrix(y_test, y_pred_lr)
ConfusionMatrixDisplay(cm_lr).plot(cmap='Greys')
plt.title("Matrice de confusion - Régression Logistique")
plt.show()
"""

#random forest 
print("=== RÉSULTATS RANDOM FOREST ===")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train) 
y_pred_rf = rf.predict(X_test)
print("\n", classification_report(y_test, y_pred_rf))
print("=== RÉSULTATS DÉTAILLÉS RANDOM FOREST (BRUTS) ===")
precision = precision_score(y_test, y_pred_rf, pos_label='Malicious')
recall = recall_score(y_test, y_pred_rf, pos_label='Malicious')
f1 = f1_score(y_test, y_pred_rf, pos_label='Malicious')
acc = accuracy_score(y_test, y_pred_rf)

print(f"Précision brute : {precision}")
print(f"Rappel brut    : {recall}")
print(f"F1-Score brut  : {f1}")
print(f"Accuracy brute  : {acc}")
"""
cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf).plot(cmap='Greys')
plt.title("Matrice de confusion - Random Forest")
plt.show()
"""
# === K-NEAREST NEIGHBORS (k-NN) ===
print("=== RÉSULTATS k-NN ===")
# On utilise généralement k=5 par défaut
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train) # Important : utiliser les données scalées
y_pred_knn = knn.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_knn))
print("=== RÉSULTATS DÉTAILLÉS K-NN (BRUTS) ===")
precision = precision_score(y_test, y_pred_knn, pos_label='Malicious')
recall = recall_score(y_test, y_pred_knn, pos_label='Malicious')
f1 = f1_score(y_test, y_pred_knn, pos_label='Malicious')
acc = accuracy_score(y_test, y_pred_knn)

print(f"Précision brute : {precision}")
print(f"Rappel brut    : {recall}")
print(f"F1-Score brut  : {f1}")
print(f"Accuracy brute  : {acc}")
"""
cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(cm_knn).plot(cmap='Greys')
plt.title("Matrice de confusion - k-NN")
plt.show()
"""
# === NAIVE BAYES (Gaussian) ===
print("=== RÉSULTATS NAIVE BAYES ===")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_nb))
print("=== RÉSULTATS DÉTAILLÉS NAIVE-BAYES (BRUTS) ===")
precision = precision_score(y_test, y_pred_nb, pos_label='Malicious')
recall = recall_score(y_test, y_pred_nb, pos_label='Malicious')
f1 = f1_score(y_test, y_pred_nb, pos_label='Malicious')
acc = accuracy_score(y_test, y_pred_nb)

print(f"Précision brute : {precision}")
print(f"Rappel brut    : {recall}")
print(f"F1-Score brut  : {f1}")
print(f"Accuracy brute  : {acc}")

"""
cm_nb = confusion_matrix(y_test, y_pred_nb)
ConfusionMatrixDisplay(cm_nb).plot(cmap='Greys')
plt.title("Matrice de confusion - Naive Bayes")
plt.show()
"""
print("=== RÉSULTATS SVM ===")

svm = SVC(kernel='rbf', class_weight="balanced", random_state=42)
svm.fit(X_train_scaled, y_train) 
y_pred_svm = svm.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_svm))
print("=== RÉSULTATS DÉTAILLÉS SVM (BRUTS) ===")
precision = precision_score(y_test, y_pred_svm, pos_label='Malicious')
recall = recall_score(y_test, y_pred_svm, pos_label='Malicious')
f1 = f1_score(y_test, y_pred_svm, pos_label='Malicious')
acc = accuracy_score(y_test, y_pred_svm)

print(f"Précision brute : {precision}")
print(f"Rappel brut    : {recall}")
print(f"F1-Score brut  : {f1}")
print(f"Accuracy brute  : {acc}")
"""
cm_svm = confusion_matrix(y_test, y_pred_svm)
ConfusionMatrixDisplay(cm_svm).plot(cmap='Greys')
plt.title("Matrice de confusion - SVM")
plt.show() 
"""