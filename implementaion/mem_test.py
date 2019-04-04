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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from Decisiontree import DecisionTreeNode
from Decisiontree_C45 import DecisionTreeC45Node
from memory_profiler import profile


@profile
def spam_dataset_process():
    # Loading dataset
    attributes = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
                  'word_freq_over', 'word_freq_remove', ' word_freq_internet', 'word_freq_order', 'word_freq_mail',
                  'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report',
                  'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email',
                  'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
                  'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
                  'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
                  'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
                  'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
                  'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu',
                  'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
                  'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
                  'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'Class']

    df = pd.read_csv("./dataset/spambase.data", header=None, names=attributes)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    str_cols = None

    return X, y, str_cols


@profile
def ckd_dataset_process():
    df = pd.read_csv("./dataset/chronic_kidney_disease_full.csv", na_values=['?', '\t?'])
    print(df.shape)
    df = df.drop(columns=['id'])
    df.head()

    # Fill NA values with most frequent values in the column
    df = df.fillna(df.mode().iloc[0])

    # In[4]:

    new_col = []
    for col in df.columns:
        col = col.replace("'", '')
        new_col.append(col)
    df.columns = new_col

    str_cols = []
    num_cols = []
    for col in df.drop(columns='class').columns:
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            str_cols.append(col)
        else:
            num_cols.append(col)

    print("categorical columns: {}\n".format(str_cols))
    print("numerical columns: {}\n".format(num_cols))

    X = df.drop(columns='class')
    y = df['class']
    return X, y, str_cols


@profile
def ckd_dataset_sklearn_process():
    df = pd.read_csv("./dataset/chronic_kidney_disease_full.csv", na_values=['?', '\t?'])
    print(df.shape)
    df = df.drop(columns=['id'])
    df.head()

    # Fill NA values with most frequent values in the column
    df = df.fillna(df.mode().iloc[0])

    # In[4]:

    new_col = []
    for col in df.columns:
        col = col.replace("'", '')
        new_col.append(col)
    df.columns = new_col

    str_cols = []
    num_cols = []
    for col in df.drop(columns='class').columns:
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            str_cols.append(col)
        else:
            num_cols.append(col)

    print("categorical columns: {}\n".format(str_cols))
    print("numerical columns: {}\n".format(num_cols))

    df_ecd = pd.get_dummies(df, columns=str_cols)
    print("{} columns: {}".format(len(df_ecd.columns), df_ecd.columns.tolist()))
    df_ecd = df_ecd.replace('ckd', 1).replace('notckd', 0)

    X = df_ecd.drop(columns='class')
    y = df_ecd['class']
    return X, y, str_cols


@profile
def bc_dataset_process():
    attributes = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                  "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin",
                  "Normal Nucleoli", "Mitoses", "Class"]
    attributes1 = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                   "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin",
                   "Normal Nucleoli", "Mitoses"]
    df = pd.read_csv("./dataset/breast-cancer-wisconsin.data", header=None, names=attributes, na_values=['?'])
    df['Class'].replace(to_replace=2, value=0, inplace=True)
    df['Class'].replace(to_replace=4, value=1, inplace=True)
    df = df.drop(columns=["Sample code number"])

    df = df.fillna(df.mode().iloc[0])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    str_cols = X.columns.tolist()

    return X, y, str_cols


@profile
def cart_classification(X_train, X_test, y_train, y_test, str_cols):
    clf_dt = DecisionTreeNode(max_depth=8, cat_features=str_cols)
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    s = accuracy_score(y_test, y_pred)
    return s


@profile
def sklearn_cart_classification(X_train, X_test, y_train, y_test, str_cols):
    clf_dt = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, min_samples_leaf=1, min_samples_split=3)
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    s = accuracy_score(y_test, y_pred)
    return s


@profile
def c45_classification(X_train, X_test, y_train, y_test, str_cols):
    clf_dt = DecisionTreeC45Node(max_depth=8, cat_features=str_cols)
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    s = accuracy_score(y_test, y_pred)
    return s


if __name__ == '__main__':
    #########################################################################
    # process data set
    # X, y, str_cols = ckd_dataset_process()
    # X, y, str_cols = bc_dataset_process()
    X, y, str_cols = spam_dataset_process()


    # X, y, str_cols = ckd_dataset_sklearn_process()

    #########################################################################
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #########################################################################
    # our implementation
    s = cart_classification(X_train, X_test, y_train, y_test, str_cols)
    # s = c45_classification(X_train, X_test, y_train, y_test, str_cols)

    # sklearn
    # s = sklearn_cart_classification(X_train, X_test, y_train, y_test, str_cols)
