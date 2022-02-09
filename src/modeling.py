from sklearn.ensemble import GradientBoostingClassifier
import pickle


def fit_model(X_train,y_train):
    model = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.1, max_features = 'sqrt', subsample = 0.5)
    
    # We weight the training to focus on important customers
    model.fit(X_train,y_train,sample_weight=X_train['mean_sales'] * X_train['n_orders'])
    save_model(model)
    return model

def save_model(model,path = 'model/model.pkl'):
    pickle.dump(model, open(path, 'wb'))
    
def load_model(path = 'model/model.pkl'):
    return pickle.load(open(path,'rb'))