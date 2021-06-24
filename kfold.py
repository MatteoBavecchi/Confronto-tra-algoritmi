import numpy as np
import pandas as pd
from numpy import mean
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import execute_test

lb_en = LabelEncoder()
sc = StandardScaler()

df = pd.read_csv('datasets/wine_quality.csv')
X = df.drop('Class', axis=1)
y = lb_en.fit_transform(df['Class'])
print("Wine quality dataset")
execute_test(X, y)

df = pd.read_csv('datasets/waveform_v2.csv')
X = df.drop('class', axis=1)
y = lb_en.fit_transform(df['class'])
print("WaveformV2 dataset")
execute_test(X, y)

df = pd.read_csv('datasets/nursery.csv')
X = df.drop('Class', axis=1)
y = lb_en.fit_transform(df['Class'])
print("Nursery dataset")
execute_test(X, y)

df = pd.read_csv('datasets/page_blocks.csv')
X = df.drop('class', axis=1)
y = lb_en.fit_transform(df['class'])
print("Page blocks dataset")
execute_test(X, y)

data = pd.read_csv('datasets/ozone.csv', header=None)
df = pd.DataFrame(data)
df = df.drop([0], axis=1)

# si sostituisce gli ? con NaN
for i in df.columns:
    df[i] = df[i].replace(['?'], np.nan)

#si converte in float i valori, che sono string
for i in df.columns[:-1]:
    df[i] = df[i].astype(str).astype(float)


#si riempie i posti con il valore NaN con la media della colonna
null_col = []
for i in df.columns:
    if df[i].isna().mean()*100 > 0:
        null_col.append(i)
for i in null_col:
    df[i] = df[i].fillna(df[i].mean())

X = df.drop([73], axis=1)
y = df[73]
X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
print("Ozone dataset")
execute_test(X, y)
