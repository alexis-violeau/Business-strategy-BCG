from sklearn.ensemble import GradientBoostingClassifier
import pickle


def get_model():
    return GradientBoostingClassifier()

def save_model(model,path = 'model/model.pkl'):
    pickle.dump(model, open(path, 'wb'))
    
def load_model(path = 'model/model.pkl'):
    return pickle.load(open(path,'rb'))