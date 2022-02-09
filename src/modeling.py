from sklearn.ensemble import GradientBoostingClassifier
import pickle


def fit_model(X_train_resample,y_train_resample):
    model = GradientBoostingClassifier(learning_rate = 0.1, subsample = 0.5, max_features = 'sqrt')
    model.fit(X_train_resample,y_train_resample,sample_weight=X_train_resample['mean_sales'] * X_train_resample['n_orders'])
    save_model(model)
    return model

def save_model(model,path = 'model/model.pkl'):
    pickle.dump(model, open(path, 'wb'))
    
def load_model(path = 'model/model.pkl'):
    return pickle.load(open(path,'rb'))