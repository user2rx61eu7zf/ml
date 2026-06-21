import pandas as pd
from urllib.parse import unquote

df = pd.read_csv("Payloads.csv")

df = df.dropna(subset=["Class"])
df = df.drop_duplicates()

# clean text
df['Payloads'] = df['Payloads'].str.encode('utf-8', errors='ignore').str.decode('utf-8')
df['Payloads'] = df['Payloads'].str.lower()
df['Payloads'] = df['Payloads'].apply(unquote)

# FIX CLASS


# FILTER
df = df[df['Payloads'].str.contains(
    r'http|script|iframe|onerror|onload|alert',
    case=False,
    na=False
)]

print(f"ligne total:{len(df)}")

malicious = df[df['Class'] == 1]
benign = df[df['Class'] == 0]
pd.set_option('display.max_colwidth', None)
print(f"Malicious : {len(malicious)}")
print(f"Benign : {len(benign)}")
print(df[df['Class'] == 1].head(10))
print(df[df['Class'] == 0].head(10))
df.to_csv("jdid.csv", index=False, encoding="utf-8")
