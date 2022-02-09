from sklearn.metrics import confusion_matrix
import seaborn as sns
from src import modeling
import matplotlib.pyplot as plt

def evaluate_model(model,X_val,y_val):
    y_pred = model.predict(X_val)
    sns.heatmap(confusion_matrix(y_val,y_pred),annot = True)
    plt.show()