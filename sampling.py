# Grey Files
# Muhammad Alahmadi
# Md Mirajul Islam

from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def one_sided_selection(x_data, y_data, random_state=0):
    """
    Resample the given data to better balance the target classes using One Sided Selection

    Input:
        x_data: attributes of the set to be resampled
        y_data: target values of the set to be resampled
        random_state: random seed for the resampling

    Output:
        X_resampled: attributes of the resampled data set
        y_resampled: target values of the resampled data set
    """
    return OneSidedSelection(random_state=random_state).fit_resample(x_data, y_data)

def random_undersampling(x_data, y_data, random_state=0):
    """
    Resample the given data to better balance the target classes using Random Undersampling

    Input:
        x_data: attributes of the set to be resampled
        y_data: target values of the set to be resampled
        random_state: random seed for the resampling

    Output:
        X_resampled: attributes of the resampled data set
        y_resampled: target values of the resampled data set
    """
    return RandomUnderSampler(random_state=random_state).fit_resample(x_data, y_data)

def smote(x_data, y_data, random_state=0):
    """
    Resample the given data to better balance the target classes using SMOTE (Synthetic Minority Over-sampling Technique)

    Input:
        x_data: attributes of the set to be resampled
        y_data: target values of the set to be resampled
        random_state: random seed for the resampling

    Output:
        X_resampled: attributes of the resampled data set
        y_resampled: target values of the resampled data set
    """
    return SMOTE(random_state=random_state).fit_resample(x_data, y_data)

def near_miss_3(x_data, y_data):
    """
    Resample the given data to better balance the target classes using our own version of the Near Miss - version 3 algorithm

    ASSUMPTIONS:
    1) Only 2 classes are present in the y_data list
    2) Exactly 4 numeric attributes are present in the x_data dataframe

    Input:
        x_data: dataframe of attributes of the set to be resampled
        y_data: list of target values of the set to be resampled

    Output:
        X_resampled: attributes of the resampled data set
        y_resampled: target values of the resampled data set
    """
    '''
    X_resampled = pd.DataFrame()
    y_resampled = []

    counts = Counter(y_data).most_common(2)
    majority_target = counts[0][0]
    minority_target = counts[1][0]

    majority_indices = [i for i in range(len(y_data)) if y_data[i] == majority_target]
    minority_indices = [i for i in range(len(y_data)) if y_data[i] == minority_target]

    x_majority = x_data.iloc[majority_indices]
    x_minority = x_data.iloc[minority_indices]

    y_majority = y_data[majority_indices]
    y_minority = y_data[minority_indices]

    X_resampled = pd.concat([X_resampled, x_minority])
    for entry in y_minority:
        y_resampled.append(entry)

    avg_majority_dist = []

    for index, maj_point in x_majority.iterrows():
        total_dist = 0.0
        for index2, min_point in x_minority.iterrows():
            total_dist += np.sqrt(np.square(maj_point.array[0]-min_point.array[0])+np.square(maj_point.array[1]-min_point.array[1])+np.square(maj_point.array[2]-min_point.array[2])+np.square(maj_point.array[3]-min_point.array[3]))
        avg_majority_dist.append(total_dist / len(y_minority))
    
    X_resampled = X_resampled.reset_index().drop('index', axis=1)
    for i in range(0, len(y_minority)):
        min_dist_index = avg_majority_dist.index(min(avg_majority_dist))
        X_resampled.loc[len(X_resampled.index)] = x_majority.iloc[min_dist_index].array
        y_resampled.append(y_majority[min_dist_index])
        x_majority.drop(min_dist_index)
        del y_majority[min_dist_index]
        del avg_majority_dist[min_dist_index]

    return X_resampled, y_resampled
    '''

    counts = Counter(y_data).most_common(2)
    majority_target = counts[0][0]
    minority_target = counts[1][0]

    majority_indices = [i for i in range(len(y_data)) if y_data[i] == majority_target]
    minority_indices = [i for i in range(len(y_data)) if y_data[i] == minority_target]

    x_majority = x_data[majority_indices]
    x_minority = x_data[minority_indices]

    y_majority = y_data[majority_indices]
    y_minority = y_data[minority_indices]

    X_resampled = x_minority
    y_resampled = y_minority

    avg_majority_dist = np.empty((0, 0))

    for maj_point in x_majority:
        total_dist = 0.0
        for min_point in x_minority:
            total_dist += np.sqrt(np.square(maj_point[0]-min_point[0])+np.square(maj_point[1]-min_point[1])+np.square(maj_point[2]-min_point[2])+np.square(maj_point[3]-min_point[3]))
        avg_majority_dist = np.append(avg_majority_dist, total_dist / len(y_minority))
    
    for i in range(0, len(y_minority)):
        min_dist_index = np.argmin(avg_majority_dist)
        X_resampled = np.append(X_resampled, [x_majority[min_dist_index]], axis = 0)
        y_resampled = np.append(y_resampled, y_majority[min_dist_index])
        x_majority = np.delete(x_majority, min_dist_index, axis = 0)
        y_majority = np.delete(y_majority, min_dist_index)
        avg_majority_dist = np.delete(avg_majority_dist, min_dist_index)

    return X_resampled, y_resampled

# Data sets to test the resampling techniques
'''
x_data = np.array([[1, 6, 1, 6], [2, 7, 2, 7], [3, 8, 3, 8], [4, 9, 4, 9], [5, 10, 5, 10]])
y_data = np.array(['False', 'False', 'False', 'True', 'True'])
X_resampled, y_resampled = near_miss_3(x_data, y_data)
print(X_resampled)
print(y_resampled)


X, y = make_classification(n_samples=10000, n_features=4, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
X_resampled2, y_resampled2 = near_miss_3(X, y)
print(Counter(y))
print(Counter(y_resampled2))

'''