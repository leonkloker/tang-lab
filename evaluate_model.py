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
    mean_error = mae(y, y_pred)
    r = pearsonr(y, y_pred)[0]
    plt.figure()
    if classes != None:
        plt.scatter(y, y_pred, c=classes, label='Mean absolute error = {:.4f}\nPearson coefficient = {:.4f}'.format(mean_error, r), s=70, 
                    marker='o', edgecolors='black')
    else:
        plt.scatter(y, y_pred, label='Mean absolute error = {:.4f}\nPearson coefficient = {:.4f}'.format(mean_error, r), s=70, 
                    marker='o', edgecolors='black')
    ax, fig = plt.gca(), plt.gcf()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.scatter(y, y, marker='.', color='red')
    plt.legend(prop={'size': 12})
    plt.plot([0, plt.gca().get_ylim()[1]], [0, plt.gca().get_ylim()[1]], '--', color='red')
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xlabel('Ground truth', fontsize=15)
    plt.ylabel('Model prediction', fontsize=15)
    plt.tight_layout()
    plt.savefig(savepath, dpi=400)

def mae(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y - y_pred))

# given the ground truth and the prediction,
# plot the confusion matrix and save the plot
def plot_confusion_matrix(y_true, y_pred, bins, savepath, labels=False, normalize=False):
    if not labels:
        y_pred = np.digitize(y_pred, bins=bins)
        y_true = np.digitize(y_true, bins=bins)
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(bins)-1)])
    if normalize:
        cm = np.array([cm[i] / np.inf if cm.sum(axis=1)[i] == 0 else cm[i] / cm.sum(axis=1, dtype=np.float32)[i] for i in range(len(cm))])
        cm = np.round(cm * 100) / 100
        cm = np.array([cm[i] / np.inf if cm.sum(axis=1)[i] == 0 else cm[i] / cm.sum(axis=1)[i] for i in range(len(cm))])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    fig, ax = plt.gcf(), plt.gca()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.xticks(ticks=np.linspace(-0.5, (len(bins)-2) + 0.5, len(bins)), labels=[f'{x:.2f}' for x in bins])
    plt.yticks(ticks=np.linspace(-0.5, (len(bins)-2) + 0.5, len(bins)), labels=[f'{x:.2f}' for x in bins])
    plt.tick_params(axis='x', direction='inout', labeltop=True, labelbottom=False)
    plt.xlabel('Model prediction', fontsize=15)
    plt.ylabel('Ground truth', fontsize=15)
    plt.tight_layout()
    plt.savefig(savepath, dpi=400)
    plt.close()
