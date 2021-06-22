from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron


def get_models():
    models = []
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('Perc', Perceptron()))
    return models


cv = RepeatedKFold(n_repeats=15, n_splits=10, random_state=50)
models = get_models()

sc = StandardScaler()


def execute_test(X, y):
    X = sc.fit_transform(X)
    for name, model in models:
        results = cross_val_score(
            model, X, y, cv=cv, scoring='accuracy', n_jobs=8)
        print('\n%s: %f +-%f\n' %
              (name, results.mean()*100, results.std()*100))
