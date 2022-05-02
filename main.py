from tokenize import Double
from turtle import color
from sklearn.model_selection import KFold, StratifiedKFold
from random import shuffle
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
'exec(%matplotlib inline)'

col_names = ["a1", "a2", "a3", "a4", "a5", "a6", "b1", "b2", "b3", "b4", "b5", "b6", "c1", "c2", "c3", "c4", "c5", "c6", "d1", "d2", "d3",
             "d4", "d5", "d6", "e1", "e2", "e3", "e4", "e5", "e6", "f1", "f2", "f3", "f4", "f5", "f6", "g1", "g2", "g3", "g4", "g5", "g6", "class"]

subset = [[40, 60], [60, 40], [80, 20], [90, 10]]

df = pd.read_csv('./connect-4.data', names=col_names)
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.describe()
X = df.drop(['class'], axis=1)
y = df['class']


# 1. Preparing the datasets

def prepareDataset(trainSize, testSize):
    return train_test_split(X, y, test_size=testSize*0.01, train_size=trainSize*0.01, shuffle=True, stratify=y)


# 2. Building the decision tree classifiers
def drawDecisionTreeClassifiers(clf, trainSize):
    figScreen = plt.figure(figsize=(15, 15))
    temp = plot_tree(clf,
                     feature_names=X.columns,
                     filled=True, rounded=True, fontsize=4)

    plt.savefig("Tree(" + str((1-0.2)*100) + "_" + str(0.2*100) + ").png")
    plt.show()

# 3. Evaluating the decision tree classifiers


def evaluatingDT(clf, feature_test, label_test):
    y_pred = clf.predict(feature_test)
    print("Decision Tree Classifier report \n",
          classification_report(label_test, y_pred))

    cfm = confusion_matrix(label_test, y_pred)
    sns.heatmap(cfm, annot=True,  linewidths=.5, cbar=None)
    plt.title('Decision Tree Classifier confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 4.The depth and accuracy of a decision tree


def drawSixDTMaxDeapth():
    max_depth_list = [1,2,3,4,5,6]
    x1 = [1,2,3,4,5,6]
    y1 = [] #x1, y1 array position test set 
    feature_train, feature_test, label_train, label_test = prepareDataset(subset[2][0], subset[2][1])
    for x in max_depth_list:
        dtc = DecisionTreeClassifier(max_depth=x) 
        # dtc.fit(feature_train,label_train)
        # drawDecisionTreeClassifiers(dtc, 80)
        dtc.fit(feature_test, label_test)
        label_predict = dtc.predict(feature_test)
        score = accuracy_score(label_test, label_predict)
        print("Accuracy score:", score)
        y1.append(score)
    plt.plot(x1, y1, color='green', linewidth = 3, marker='o')
    plt.title("Decision Tree Accuracy vs Depth of tree")
    plt.xlabel("Depth of tree")
    plt.ylabel("Accuracy")
    plt.show()


def test_(clf, feature_test, label_test):  
    label_predict = clf.predict(feature_test)
    print("Accuracy score:", accuracy_score(label_test, label_predict))


def let_user_pick(options):
    print("Please choose:")
    for idx, element in enumerate(options):
        print("{}) {}".format(idx+1, element))
    i = input("Enter number: ")
    try:
        if 1 < int(i) <= len(options):
            return int(i)
    except:
        pass
    return None


def main():
    options = ["Prepare dataset(auto prepare).", "Building the decision tree classifiers.",
               "Evaluating the decision tree classifiers", "The depth and accuracy of a decision tree"]

    feature_train, feature_test, label_train, label_test = prepareDataset(
        subset[2][0], subset[2][1])

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(feature_train, label_train)

    choice = let_user_pick(options)
    if(choice == 2):
        drawDecisionTreeClassifiers(clf, subset[2][0])
    if(choice == 3):
        evaluatingDT(clf, feature_test, label_test)
    if(choice == 4):
        drawSixDTMaxDeapth()
        # test_(clf, feature_test, label_test)
        # depthAndAccuracy()
        # else:
        #     check = False


if __name__ == '__main__':
    main()
