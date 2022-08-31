from tpot import TPOTClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.preprocessing import OneHotEncoder

BDD = pd.read_excel('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/2-data_ossements/2-2 custom/feature_table_mzwid_0.019_minfrac_0_no_int.xlsx')
BDD2 = BDD
del BDD2['label']
y = BDD2['feature']
X = BDD2.drop('feature', axis=1)
y, uniques = pd.factorize(y) # on appel une autre variale afin d'y inclure l'Index
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0, stratify=y)


tpot = TPOTClassifier(n_jobs=-1, verbosity=3,  max_time_mins=900)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('C:/Users/orkad/Desktop/stage emmanuel/travail/code python/tpot_pipeline.py')
print(tpot.evaluated_individuals_)
