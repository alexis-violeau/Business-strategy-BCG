from imblearn.under_sampling import RandomUnderSampler

def resample_data(X_train,y_train,ratio = 0.1):
    return RandomUnderSampler(sampling_strategy = ratio).fit_resample(X_train,y_train)