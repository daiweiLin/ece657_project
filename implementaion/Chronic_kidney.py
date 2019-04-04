
# coding: utf-8

# In[1]:


import numpy as np  
import matplotlib.pyplot as plt 
import os
import gc
import psutil
import pandas as pd  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import collections
from sklearn.tree import export_graphviz
import seaborn as sns   
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import matplotlib.gridspec as gridspec
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

df = pd.read_csv("./dataset/chronic_kidney_disease_full.csv", na_values=['?','\t?'])
print(df.shape)
df = df.drop(columns = ['id'])
df.head()


# Fill NA values with most frequent values in the column
df = df.fillna(df.mode().iloc[0])



# In[4]:


new_col = []
for col in df.columns:
    col = col.replace("'",'')
    new_col.append(col)
df.columns = new_col


str_cols = []
num_cols = []
for col in df.drop(columns = 'class').columns:
    if df[col].dtype != np.int64 and df[col].dtype != np.float64:
        str_cols.append(col)
    else:
        num_cols.append(col)
    
print("categorical columns: {}\n".format(str_cols))
print("numerical columns: {}\n".format(num_cols))



# ### Optimal Tree

# # CART


from sklearn.metrics import precision_recall_fscore_support
from Decisiontree import DecisionTreeNode


def find_best_clf(n, X, y, classifier, parameters, labels, cv=5):

    best_para = []
    best_tree = []
    train_score = []
    test_score = []
    train_time = []
    classification_time = []
    
    precision = []
    recall = []
    fmeasure = []
    
    for i in range(n):
        print("Iteration {}".format(i))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        gridsearch = GridSearchCV(clf_dt, parameters, cv=cv, return_train_score=True)
        gridsearch.fit(X_train, y_train)
        grid_search_result = pd.DataFrame(gridsearch.cv_results_ )
        
        optimal_tree = gridsearch.best_estimator_
        
        ###########################################################
        
        best_para.append(gridsearch.best_params_)
        best_tree.append(optimal_tree)
        test_score.append(optimal_tree.score(X_test, y_test))
        train_score.append(gridsearch.cv_results_['mean_train_score'][gridsearch.best_index_])
        classification_time.append(gridsearch.cv_results_['mean_score_time'][gridsearch.best_index_])
        train_time.append(gridsearch.cv_results_['mean_fit_time'][gridsearch.best_index_])
        
        y_pred = optimal_tree.predict(X_test)
        pr, r, fs, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, labels=labels)
        precision.append(pr)
        recall.append(r)
        fmeasure.append(fs)
        
    return best_para, best_tree, train_score, test_score, train_time, classification_time, precision, recall, fmeasure


# In[14]:


# X = df_ecd.drop(columns='class')
# y = df_ecd['class']

# parameters = {"max_depth": [1, 5, 8, 10, 20, 100, 1000, 10000],
#               "min_samples_split": [2, 3, 5, 10, 100],
#               "min_samples_leaf": [1, 5, 10, 100, 1000, 10000],
#               "max_leaf_nodes": [None, 10, 100, 1000, 10000],
#               }

X = df.drop(columns='class')
y = df['class']
parameters = {"max_depth": [1, 5, 8, 10, 20, 100, 1000]}

labels = ["ckd","notckd"]

clf_dt = DecisionTreeNode(max_depth=1, cat_features=str_cols)
best_para, best_tree, train_score, test_score, train_time, classification_time ,precision, recall, fmeasure= find_best_clf(10, X, y, clf_dt, parameters, labels)


# In[15]:


for stats, name in zip([train_score, test_score, train_time, classification_time], ["train_score", "test_score", "train_time", "classification_time"]):
    print("Average {}: {}".format(name, np.average(stats)))


# In[31]:


(1-0.946969696969697)*df.shape[0]*0.33


# In[25]:


for stats, name in zip([precision, recall, fmeasure], ["precision", "recall", "fmeasure"]):
    avg = np.average(stats, axis = 0)
    print("Average ckd {}: {}".format(name, avg[0]))
    print("Average notckd {}: {}".format(name, avg[1]))


# In[17]:


print(max(test_score))
idx = test_score.index(max(test_score))
print("best parameters: {}".format(best_para[idx]))


# In[34]:


best_tree[idx].print_tree()


# In[62]:


dot_data = StringIO()
export_graphviz(best_tree[5], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['ckd','notckd'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.set_size('"40,40!"')
graph.write_png('Optimal_CART_chronic_kidney.png')
Image(graph.create_png())


# In[57]:


corr_matrix = np.absolute(df_ecd.astype(float).corr())
corr_matrix.loc['class'].sort_values(ascending=False)


# In[46]:


plt.figure(figsize=(10,10))
sns.heatmap(np.absolute(df_ecd[['sg', 'pcv', 'hemo', 'htn_no', 'htn_yes', 'dm_no', 'dm_yes','al', 'rbcc',]].astype(float).corr()),linewidths=1.0,vmax=3.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[53]:


corr_matrix = np.absolute(df_ecd.astype(float).corr())
corr_matrix.loc['sc'].sort_values(ascending=False)


# # Use C4.5

# In[27]:


from implementaion.Decisiontree_C45 import DecisionTreeC45Node
X = df.drop(columns='class')
y = df['class']
parameters = {"max_depth": [1, 5, 8, 10, 20, 100, 1000]}

labels = ["ckd","notckd"]

clf_dt = DecisionTreeC45Node(max_depth=1, cat_features=str_cols)
best_para, best_tree, train_score, test_score, train_time, classification_time ,precision, recall, fmeasure= find_best_clf(10, X, y, clf_dt, parameters, labels)


# In[28]:


for stats, name in zip([train_score, test_score, train_time, classification_time], ["train_score", "test_score", "train_time", "classification_time"]):
    print("Average {}: {}".format(name, np.average(stats)))


# In[30]:


(1-0.9810606060606061)*df.shape[0]*0.33


# In[29]:


for stats, name in zip([precision, recall, fmeasure], ["precision", "recall", "fmeasure"]):
    avg = np.average(stats, axis = 0)
    print("Average ckd {}: {}".format(name, avg[0]))
    print("Average notckd {}: {}".format(name, avg[1]))


# -----------------------------------------------------------------

# ### Default tree

# In[72]:


import time
from sklearn.metrics import precision_recall_fscore_support
def repeat_clf_n_times(n, X, y, classifier, labels):

    trees = []
    train_score = []
    test_score = []
    train_time = []
    classification_time = []
    
    precision = []
    recall = []
    fmeasure = []
    
    
    for i in range(n):
        print("Iteration {}".format(i))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        start_time = time.time()
        classifier.fit(X_train, y_train)
        tr_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        cls_time = time.time() - start_time
        
        trees.append(classifier)
        train_score.append(classifier.score(X_train, y_train))
        test_score.append(classifier.score(X_test, y_test))
        
        train_time.append(tr_time)
        classification_time.append(cls_time)
        
        
        pr, r, fs, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, labels=labels)
        precision.append(pr)
        recall.append(r)
        fmeasure.append(fs)
        
    return trees, train_score, test_score, train_time, classification_time, precision, recall, fmeasure


# In[73]:


clf = DecisionTreeClassifier()
labels = ["ckd","notckd"]
trees, train_score, test_score, train_time, classification_time, precision, recall, fmeasure = repeat_clf_n_times(10, X, y, clf, labels)


# In[74]:


for stats, name in zip([train_score, test_score, train_time, classification_time], ["train_score", "test_score", "train_time", "classification_time"]):
    print("Average {}: {}".format(name, np.average(stats)))


# In[65]:


(1-0.9628787878787879) * df.shape[0] * 0.33


# In[75]:


for stats, name in zip([precision, recall, fmeasure], ["precision", "recall", "fmeasure"]):
    print("Average ckd {}: {}".format(name, np.average(stats[0])))
    print("Average notckd {}: {}".format(name, np.average(stats[0])))

