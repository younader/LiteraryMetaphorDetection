
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import numpy as np

def plot_hbar(data):
  """
    Plots Horizontal Bar Graph for the data object
    Args:
        data: (dict) of string keys and number values
  """
  names = list(data.keys())
  values = list(data.values())
  fig, axs = plt.subplots(1, 1, figsize=(8, 8), sharey=True)
  axs.barh(names, values)
  # plt.xticks(rotation='45')
  plt.margins(0.2)
  plt.subplots_adjust(bottom=0.15)
def plot_auc_cv(classifier,X,y,n_folds=10):

  """
    Plots the cross-validated ROC AUC for the classifier on the
    provided data
    Args:
        classifier: (object) that implements predict_proba and fit functions
        X: (n.darray,n.darray) feature vector
        y: (n.darray) labels
        folds: (int) number of cross validation folds
  """
  n_samples, n_features = X.shape
  cv = StratifiedKFold(n_splits=n_folds)
  # classifier = svm.SVC(kernel="rbf", probability=True)
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)

  fig, ax = plt.subplots(figsize=(8,8))
  for i, (train, test) in enumerate(cv.split(X, y)):
      classifier.fit(X[train], y[train])
      viz = plot_roc_curve(classifier, X[test], y[test],
                          name='ROC fold {}'.format(i),
                          alpha=0.3, lw=1, ax=ax)
      interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
      interp_tpr[0] = 0.0
      tprs.append(interp_tpr)
      aucs.append(viz.roc_auc)

  ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(
      mean_fpr,
      mean_tpr,
      color="b",
      label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (np.mean(aucs), np.std(aucs)),
      lw=2,
      alpha=0.8,
  )

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(
      mean_fpr,
      tprs_lower,
      tprs_upper,
      color="grey",
      alpha=0.2,
      label=r"$\pm$ 1 std. dev.",
  )

  ax.set(
      xlim=[-0.05, 1.05],
      ylim=[-0.05, 1.05],
      title="Receiver operating characteristic example",
  )
  ax.legend(loc="lower right")
  plt.show()