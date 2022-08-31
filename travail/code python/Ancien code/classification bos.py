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
from sklearn.preprocessing import normalize

# =============================================================================
# Importation bdd
# =============================================================================

df = pd.read_csv('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/table_bos.csv')
del df['Unnamed: 0']
y = df['Label']
X = df.drop('Label', axis=1)
#X = X.astype(bool).astype(int)


# =============================================================================
# split de la bdd
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, stratify=y)
print(X_train.shape)
print(X_test.shape)

# =============================================================================
# Random forest
# =============================================================================

clf = RandomForestClassifier(n_estimators=500, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("f-mesure:", metrics.f1_score(y_test, y_pred, average='micro'))
print(metrics.classification_report(y_test, y_pred))


# =============================================================================
# mlpclassifier
# =============================================================================
from sklearn.neural_network import MLPClassifier

exported_pipeline = MLPClassifier(alpha=0.001, learning_rate_init=0.001)
exported_pipeline.fit(X_train, y_train)
y_pred = exported_pipeline.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("f-mesure:", metrics.f1_score(y_test, y_pred, average='micro'))
print(metrics.classification_report(y_test, y_pred))

# =============================================================================
# LinearSVC
# =============================================================================

from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, stratify=y)
exported_pipeline = LinearSVC(C=10.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.001)
exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, results))
print("f-mesure:", metrics.f1_score(y_test, results, average='micro'))
print(metrics.classification_report(y_test, results))

