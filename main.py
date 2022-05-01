from random import shuffle
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
'exec(%matplotlib inline)'

df = pd.read_csv('./connect-4.data')

labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])

df.describe()

#1. Preparing the datasets
X = df.drop(['class'], axis=1)
Y = df['class']

feature_train, feature_test, label_train, label_test = train_test_split(
    X, Y, test_size=0.1, shuffle=True)


#2. Building the decision tree classifiers
def drawDecisionTreeClassifiers(feature_train, label_train):
    clf = DecisionTreeClassifier()
    clf.fit(feature_train, label_train)
    dot_data = export_graphviz(clf,
                               feature_names=feature_train.columns,
                               filled=True, rounded=True,
                               special_characters=True,
                               out_file=None,
                               )
    graph = graphviz.Source(dot_data)
    graph


drawDecisionTreeClassifiers(feature_train, label_train)