import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

#random forest 
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train) 
y_pred_rf = rf.predict(X_test)
print("\n", classification_report(y_test, y_pred_rf))

# === K-NEAREST NEIGHBORS (k-NN) ===
print("=== RÉSULTATS k-NN ===")
# On utilise généralement k=5 par défaut
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train) # Important : utiliser les données scalées
y_pred_knn = knn.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_knn))

# === NAIVE BAYES (Gaussian) ===
print("=== RÉSULTATS NAIVE BAYES ===")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_nb))

print("=== RÉSULTATS SVM ===")

svm = SVC(kernel='rbf', class_weight="balanced", random_state=42)
svm.fit(X_train_scaled, y_train) 
y_pred_svm = svm.predict(X_test_scaled)
print("\n", classification_report(y_test, y_pred_svm))