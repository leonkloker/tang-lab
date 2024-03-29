import joblib
import numpy as np
import pandas as pd
import sklearn.linear_model as sklin
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# given the ground truth and the prediction, 
# plot the ground truth vs the prediction and save the plot
def plot_prediction(y, y_pred, savepath, title=None):
    plt.figure()
    plt.scatter(y, y_pred, label='MAE: {}'.format(title))
    plt.scatter(y, y, marker='x', color='red')
    plt.legend()
    plt.plot([0, 1], [0, 1], color='red')
    plt.grid()
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.ylim(-0.2, 1.2)
    plt.savefig(savepath)

# given the ground truth and the prediction,
# plot the confusion matrix and save the plot
def plot_confusion_matrix(y_true, y_pred, bins, savepath, labels=False):
    if not labels:
        y_pred = np.digitize(y_pred, bins=bins)
        y_true = np.digitize(y_true, bins=bins)
    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(ticks=np.linspace(-0.5, (len(bins)-2) + 0.5, len(bins)), labels=[f'{x:.2f}' for x in bins])
    plt.yticks(ticks=np.linspace(-0.5, (len(bins)-2) + 0.5, len(bins)), labels=[f'{x:.2f}' for x in bins])
    plt.tick_params(axis='x', direction='inout', labeltop=True, labelbottom=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(savepath)
