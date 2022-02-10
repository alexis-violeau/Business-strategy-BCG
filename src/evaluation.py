from sklearn.metrics import confusion_matrix
import seaborn as sns
from src import modeling
import matplotlib.pyplot as plt

def evaluate_model(model,X_val,y_val):
    plt.style.use('dark_background')
    y_pred = model.predict(X_val)
    plt.title('Confusion matrix')
    sns.heatmap(confusion_matrix(y_val,y_pred,normalize = 'all'),annot = True,cmap="YlGn")
    plt.xlabel('Prediction')
    plt.ylabel('Ground truth')
    plt.xticks([0.5,1.5],['No churn','churn'])
    plt.yticks([0.5,1.5],['No churn','churn'])
    plt.show()