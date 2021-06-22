# Confronto tra algoritmi

Progetto di Intelligenza Artificiale che confronta i tre algoritmi **Naive Bayes**, **Random Forest** e **Perceptron**, su 6 datasets diversi, utilizzando la metodologia sperimentale usata nell' articolo di [Zhang & Suganthan (2014)](https://doi.org/10.1016/j.patcog.2014.04.001).

## Installazione

1. Predisporre un ambiente di sviluppo Python (e.g. [Conda](https://conda.io))
2. Installare tramite package manager Conda le dipendenze necessarie:
    1. Installare la libreria scientifica Scikit Learn `conda install -c conda-forge scikit-learn`
    2. Installare la libreria Pandas `conda install -c conda-forge pandas`
    3. Installare la libreria numpy `conda install -c conda-forge numpy`
    4. Installare la libreria Seaborn `conda install -c conda-forge numpy`

## Datasets

I datasets sono reperibili a questi link:

* [Nursery](https://www.openml.org/d/1568)
* [Letter](https://www.openml.org/d/977)
* [Ozone](https://www.kaggle.com/prashant111/ozone-level-detection) (File `eighthr.data.csv` )
* [Waveform](https://www.openml.org/d/60)
* [PageBlocks](https://datahub.io/machine-learning/page-blocks)
* [Wine quality (white)](https://www.openml.org/d/40498)

## Scripts

Tra i 6 datasets utilizzati, 5 sono stati divisi con la tecnica **k-fold cross validation**, e uno tramite divisione standard tra training set e test set. Quindi sono presenti due file script, uno per tipo di divisione:

* `kfold.py` - Datasets nursery, ozone, waveform, pageblocks e wine quality
* `letters.py` - Dataset letters
* `config.py` - Contiene le funzioni per stimare lo score dei classifiers


## Fonti utilizzate

Per il codice, oltre che a consultare molto la documentazione ufficiale di Sklearn, ho ripreso parti di codice da:

* [Air Quality Prediction - Pushkar Jain](https://www.kaggle.com/l33tc0d3r/air-quality-prediction) - per manipolare il dataset ozone prima di sottoporlo agli algoritmi
* [Wine Quality Kfold cross validation and Prediction - Suveesh](https://www.kaggle.com/suveesh/wine-quality-kfold-cross-validation-and-prediction)

## License
[MIT](https://choosealicense.com/licenses/mit/)