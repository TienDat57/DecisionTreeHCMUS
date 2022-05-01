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

df = pd.read_csv('./mushrooms.csv')
# print(df.columns)
# print(df.head(5))

labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.describe()

X = df.drop(['class'], axis=1)
Y = df['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

dot_data = export_graphviz(clf, out_file=None, 
                         feature_names=X.columns,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

features_list = X.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(5,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
# plt.show()

y_pred=clf.predict(X_test)
print("Decision Tree Classifier report \n", classification_report(Y_test, y_pred))