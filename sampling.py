from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def one_sided_selection(x_data, y_data, random_state):
    return OneSidedSelection(random_state=random_state).fit_resample(x_data, y_data)

def random_undersampling(x_data, y_data, random_state):
    return RandomUnderSampler(random_state=random_state).fit_resample(x_data, y_data)

def smote(x_data, y_data, random_state):
    return SMOTE(random_state=random_state).fit_resample(x_data, y_data)