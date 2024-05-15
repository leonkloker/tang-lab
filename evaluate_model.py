import joblib
import numpy as np
import pandas as pd
import sklearn.linear_model as sklin
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# given the ground truth and the prediction, 
# plot the ground truth vs the prediction and save the plot
def plot_prediction(y, y_pred, savepath, classes=None):
    mae = np.mean(np.abs(y - y_pred))
    r = pearsonr(y, y_pred)[0]
    plt.figure()
    if classes != None:
        plt.scatter(y, y_pred, c=classes, label='Mean absolute error = {:.4f}\nPearson coefficient = {:.4f}'.format(mae, r), s=70, 
                    marker='o', edgecolors='black')
    else:
        plt.scatter(y, y_pred, label='Mean absolute error = {:.4f}\nPearson coefficient = {:.4f}'.format(mae, r), s=70, 
                    marker='o', edgecolors='black')
    plt.scatter(y, y, marker='.', color='red')
    plt.legend(prop={'size': 12})
    plt.plot([0, plt.gca().get_ylim()[1]], [0, plt.gca().get_ylim()[1]], '--', color='red')
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xlabel('Ground truth', fontsize=12)
    plt.ylabel('Model prediction', fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=400)

# given the ground truth and the prediction,
# plot the confusion matrix and save the plot
def plot_confusion_matrix(y_true, y_pred, bins, savepath, labels=False, normalize=True):
    if not labels:
        y_pred = np.digitize(y_pred, bins=bins)
        y_true = np.digitize(y_true, bins=bins)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, len(bins)))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] if normalize else cm    
    cm = np.round(cm * 100) / 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(ticks=np.linspace(-0.5, (len(bins)-2) + 0.5, len(bins)), labels=[f'{x:.2f}' for x in bins])
    plt.yticks(ticks=np.linspace(-0.5, (len(bins)-2) + 0.5, len(bins)), labels=[f'{x:.2f}' for x in bins])
    plt.tick_params(axis='x', direction='inout', labeltop=True, labelbottom=False)
    plt.xlabel('Model prediction', fontsize=12)
    plt.ylabel('Ground truth', fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=400)
