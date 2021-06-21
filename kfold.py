import numpy as np
import pandas as pd 

import seaborn as sns
from numpy import mean

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron


def get_models():
    models = []
    models.append(('NB', GaussianNB()))
    models.append(('RF',RandomForestClassifier()))
    models.append(('Perc',Perceptron()))
    return models

cv = RepeatedKFold(n_repeats = 15, n_splits=10, random_state=50)
models = get_models()

lb_en  = LabelEncoder()
sc = StandardScaler()

#WINE QUALITY DATASET
df = pd.read_csv('wine_quality.csv')

X = df.drop('Class', axis = 1)
y = lb_en.fit_transform(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

names = []

print("Wine quality dataset")
for name, model in models:
    results =  cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy') 
    print('\n%s: %f +-%f\n' % (name, results.mean()*100, results.std()*100)) 


#WAVEFORM DATASET
df = pd.read_csv('waveform_v2.csv')

X = df.drop('class', axis = 1)
y = lb_en.fit_transform(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

names = []

print("WaveformV2 dataset")

for name, model in models:
    results =  cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy') 
    print('\n%s: %f +-%f\n' % (name, results.mean()*100, results.std()*100)) 



#NURSERY DATASET
df = pd.read_csv('nursery.csv')

X = df.drop('Class', axis = 1)
y = lb_en.fit_transform(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

names = []

print("Nursery dataset")

for name, model in models:
    results =  cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy') 
    print('\n%s: %f +-%f\n' % (name, results.mean()*100, results.std()*100)) 


#PAGEBLOCKS DATASET
df = pd.read_csv('page_blocks.csv')

X = df.drop('class', axis = 1)
y = lb_en.fit_transform(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

names = []

print("Page blocks dataset")

for name, model in models:
    results =  cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy') 
    print('\n%s: %f +-%f\n' % (name, results.mean()*100, results.std()*100)) 




#OZONE DATASET
data = pd.read_csv('ozone.csv', header=None)

df = pd.DataFrame(data)

df = df.drop([0], axis = 1)
#si sostituisce gli ? con nan
for i in df.columns:
  df[i] = df[i].replace(['?'], np.nan)

for i in df.columns[:-1]:
  df[i] = df[i].astype(str).astype(float)

null_col = []
for i in df.columns:
  if df[i].isna().mean()*100 > 0:
    null_col.append(i)
    
    
for i in null_col:
  df[i] = df[i].fillna(df[i].mean())

X = df.drop([73], axis=1)
y = df[73]
X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

names = []

print("Ozone dataset")

for name, model in models:
    results =  cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy') 
    print('\n%s: %f +-%f\n' % (name, results.mean()*100, results.std()*100)) 
