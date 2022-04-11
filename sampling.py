# Grey Files
# Muhammad Alahmadi
# Md Mirajul Islam

from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def one_sided_selection(x_data, y_data, random_state):
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

def random_undersampling(x_data, y_data, random_state):
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

def smote(x_data, y_data, random_state):
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