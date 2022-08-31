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
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2



# =============================================================================
# Importation bdd
# =============================================================================

BDD = pd.read_excel('C:/Users/Admin/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/2-data_ossements/2-2/feature_table_mzwid_0.019_minfrac_0.1_no_int.xlsx')
BDD2= BDD
del BDD2['feature']
y = BDD2['label']
X = BDD2.drop('label',axis=1)
print(X)
# =============================================================================
# split de la bdd
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(X_train.shape)
print(X_test.shape)
print(X_train)
 
# =============================================================================
# Random forest
# =============================================================================

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f-mesure:",metrics.f1_score(y_test, y_pred, average='micro'))
print(metrics.classification_report(y_test, y_pred))

# =============================================================================
# test plt
# =============================================================================
clf.feature_names = list(X_train.columns.values)
clf.feature_names
plt.figure(figsize=(480,20))
plt.bar([i for i in range(len(clf.feature_importances_))], clf.feature_importances_, tick_label=clf.feature_names)
plt.savefig('C:/Users/orkad/Desktop/stage emmanuel/image graph/test.svg')
plt.show()

# =============================================================================
# selection de variable basé sur la variance
# =============================================================================

selector = VarianceThreshold(threshold=0.05) 
selector.fit_transform(X)
selector.get_support()
np.array(clf.feature_names)[selector.get_support()]

max(X.var(axis=0))

# =============================================================================
# chi2
# =============================================================================

chi2(X, y)
selector2 = SelectKBest(chi2, k=1)
selector2.fit(X, y)
selector2.get_support()
np.array(clf.feature_names)[selector2.get_support()]
