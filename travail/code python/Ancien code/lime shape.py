#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lime
import lime.lime_tabular
from __future__ import print_function
np.random.seed(1)
import shap


# In[2]:


df = pd.read_csv('C:/Users/orkad/Desktop/stage emmanuel/datasets/table final pour ml/table_cervus.csv')
df = df.rename(columns = {'Unnamed: 0': 'label'})
df = df.fillna(0)


n = len(df.index)

x = 0
while x < n:
    a = 'cerf' in df.iloc[x,0]
    b = 'daim' in df.iloc[x,0]
    c = 'meagC' in df.iloc[x,0]
    if a == True :
        df.iloc[x,0] = 'cerf'
    if b == True :
        df.iloc[x,0] = 'daim'
    if c == True :
        df.iloc[x,0] = 'meagC' 
    x = x+1
df    

shap.initjs()


# In[3]:


y = df['label']
X = df.drop('label', axis=1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0, stratify=y)


# In[4]:


clf = RandomForestClassifier(n_estimators=500, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("f-mesure:", metrics.f1_score(y_test, y_pred, average='micro'))
print(metrics.classification_report(y_test, y_pred))
metrics.accuracy_score(y_test, y_pred)


# In[5]:


classe = df.drop('label', axis=1)
classe = classe.columns.tolist()
feature = clf.classes_
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=classe, class_names=clf.classes_, discretize_continuous=True)


# In[6]:


i = np.random.randint(0, X_test.shape[0])
i
exp = explainer.explain_instance(X_test[4], clf.predict_proba, num_features=20, top_labels=3)


# In[7]:


exp.show_in_notebook(show_table=True, show_all=False)


# In[8]:


explainer = shap.KernelExplainer(clf.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

