from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_names = ['class_' + str(i) for i in classes]

def plot_confusion_matrix(cm, classes, path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)


def save_confusion_matrix(oof_preds: pd.DataFrame, y: pd.Series, path: str):
    unique_y = np.unique(y)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i

    y_map = np.array([class_map[val] for val in y])

    cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds, axis=-1))

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(12,12))
    plot_confusion_matrix(cnf_matrix, classes=class_names, path=path, normalize=True, title='Confusion matrix')