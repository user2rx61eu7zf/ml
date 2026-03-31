import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv("nv.csv")
#print(df.Class.value_counts())
df = df.dropna(subset=["Class"])
Payload_train , Payload_test , Class_train , Class_test = train_test_split(
    df.Payloads ,
    df.Class,
    test_size=0.2,
    stratify=df.Class, 
    random_state=42)

print("nbr instance train1",Payload_train.shape,)
print("nb instace test1",Payload_test.shape)



vectorizer = TfidfVectorizer(
    ngram_range=(1,2),    # uni-grams + bi-grams
    max_features=5000,   # garder seulement 10k mots ou séquences les plus fréquentes
)
# Fit sur le train
tfidf_train = vectorizer.fit_transform(Payload_train)
# Transform sur le test
tfidf_test = vectorizer.transform(Payload_test)

df_tfidf_train = pd.DataFrame(tfidf_train.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf_train.to_csv("TFidf.csv",index=False)
df_tfidf_test = pd.DataFrame(tfidf_test.toarray(), columns=vectorizer.get_feature_names_out())
'''
top_words = df_tfidf_train.iloc[0].sort_values(ascending=False).head(100)
#print(top_words)

idf_values = vectorizer.idf_  # tableau numpy

# Associer chaque mot à son IDF
idf_dict = dict(zip(vectorizer.get_feature_names_out(), idf_values))
idf_sorted = dict(sorted(idf_dict.items(), key=lambda x: x[1], reverse=True))

# Afficher tous les tokens du plus rare au plus fréquent
# Transformer le dictionnaire en DataFrame
df_idf = pd.DataFrame(list(idf_dict.items()), columns=["token", "idf"])

# Trier du plus haut IDF au plus bas
df_idf = df_idf.sort_values(by="idf", ascending=False).reset_index(drop=True)

# Sauvegarder dans un CSV
df_idf.to_csv("idf_tokens.csv", index=False)

print("CSV 'idf_tokens.csv' créé avec tous les tokens triés par IDF !")'''