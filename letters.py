import numpy as np
from sklearn.datasets import fetch_openml
from sklearn import metrics

from config import get_models

X, y = fetch_openml('letter', version=1, return_X_y=True)
y = np.array(y)
divider = 15000
r_length = 20000

X_train = X[:divider]
X_test = X[divider:r_length]
y_train = y[:divider]
y_test = y[divider:r_length]


models = get_models()
names = []
for name, model in models:
    total=[]
    for index in range(15):
        y_pred = model.fit(X_train, y_train).predict(X_test)
        acc =  metrics.accuracy_score(y_test,y_pred)
        names.append(name)
        total.append(acc)
        print('[%f] %s: %f' % (index, name, acc*100)) 
    nptotal = np.asarray(total)
    print('\n[Media]%s: %f +-%f\n' % (name, nptotal.mean()*100, nptotal.std()*100))