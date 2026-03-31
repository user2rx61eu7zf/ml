import pandas as pd
from urllib.parse import unquote

df=pd.read_csv("Payloads.csv")
print(df.head())
df = df.dropna(subset=["Class"])
df = df.drop_duplicates()

df['Payloads'] = df['Payloads'].str.encode('utf-8', errors='ignore').str.decode('utf-8')
df['Payloads']=df['Payloads'].str.lower()

df['Class']=df['Class'].map({
    'Malicious':1,
    'Benign':0
})

df['Payloads'] = df['Payloads'].apply(unquote) #decodage url percent coding 
df = df[df['Payloads'].str.contains(r'http|scrpit|iframe|onerror|onload|alert<|>', regex=True)] #supprime li machi url wela injection kaynin li simple text

print(f"ligne total:{len(df)}")
malicious = df[df['Class'] == 1]
benign = df[df['Class'] == 0]

# Afficher le nombre de lignes de chaque
print(f"Nombre de lignes Malicious : {len(malicious)}")
print(f"Nombre de lignes Benign : {len(benign)}\n")
df.to_csv("nv.csv",index=False,encoding="utf-8")