# Grey Files
# Muhammad Alahmadi

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import csv
import pandas as pd

def get_raw_data():
    """
    Read the raw data from the source files located in the ECONet directory

    Input: None

    Output:
        train: panda containing all of the training data including target values
        test: panda containing all of the testing features without target values
    """
    file1 = open('ECONet/train.csv')
    file2 = open('ECONet/test.csv')
    csvreader1 = csv.reader(file1)
    csvreader2 = csv.reader(file2)
    header1 = []
    header2 = []
    header1 = next(csvreader1)
    header2 = next(csvreader2)
    rows1 = []
    rows2 = []
    for row in csvreader1:
        rows1.append(row)
    for row in csvreader2:
        rows2.append(row)
    train = pd.DataFrame(data=rows1, columns=header1)
    test = pd.DataFrame(data=rows2, columns=header2)
    
    return train, test

def get_data():
    """
    Get the data from the ECONet dataset in useable format

    Input: None

    Output:
        train_x: panda containing all of the attributes of the training data set
        train_y: panda containing all of the target values of the training data set
        test_x: panda containing all of the attributes of the test data set
    """
    train, test_x = get_raw_data()
    train.drop(['Station', 'Ob', 'value', 'measure'], inplace=True, axis=1)
    test_x.drop(['Station', 'Ob', 'value', 'measure'], inplace=True, axis=1)
    train_y = train['target']
    train_x = train.drop('target', axis=1)
    return train_x, train_y, test_x


def train_adaboost_classifier(x_data, y_data, n_estimators, random_state):
    """
    Train an AdaBoost classifier on the given data

    Input:
        x_data: attributes used to train the classifier
        y_data: target values used to train the classifier
        n_estimators: max number of estimators used in the AdaBoost classifier
        random_state: random seed for the classifier
    
    Output:
        trained AdaBoost classifier
    """
    return AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state).fit(x_data, y_data)

def train_svm_classifier(x_data, y_data, C, random_state):
    """
    Train an SVM classifier on the given data

    Input:
        x_data: attributes used to train the classifier
        y_data: target values used to train the classifier
        C: regularization parameter for the classifier
        random_state: random seed for the classifier
    
    Output:
        trained SVM classifier
    """
    return SVC(C=C, random_state=random_state).fit(x_data, y_data)


#Test code for reading the data values and printing them
train_x, train_y, test_x = get_data()
print(train_x)
print(train_y)
print(test_x)