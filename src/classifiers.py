from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,auc,roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from rgf.sklearn import RGFClassifier
from tqdm.notebook import tqdm
import numpy as np
from pytorch_tabnet.tab_model  import TabNetClassifier
import torch

def tabnet_auc(X,y,n_folds=10):
  """
      Benchmark Tabnet on the provided Data
      Args:
          X: (n.darray,n.darray) feature vector
          y:  (n.darray) labels
          n_folds: (int) number of cross validation folds
  """
  skf = StratifiedKFold(n_splits=10)
  results=[]
  feature_importances=[]
  for train_index, test_index in tqdm(skf.split(X, y)):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      scl=StandardScaler()
      X_train=scl.fit_transform(X_train)
      X_test=scl.transform(X_test)
      clf=TabNetClassifier(n_d=4,n_a=4,n_steps=4,
                        optimizer_params=dict(lr=3e-2),
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        scheduler_params={ "step_size":30, "gamma":0.6},
                        verbose=False
                      )
      clf.fit(X_train,y_train,batch_size=4,virtual_batch_size=8,max_epochs=100)
      results.append(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
      feature_importances.append(clf.feature_importances)
  print("Tabnet AUC:", np.mean(results))
  return results,np.mean(feature_importances,axis=0)
def classifiers_auc(X,y,n_folds=10):
  """
    Benchmarks different classifiers on the provided data
  """
  skf = StratifiedKFold(n_splits=n_folds)

  models=[("SVM",SVC(kernel='rbf',probability=True)),
          ('Naive Bayes',GaussianNB()),
          ('Logistic Regression',LogisticRegressionCV(max_iter=2000)),
          ('Knn',KNeighborsClassifier(n_neighbors=3)), 
          ('Gradient Boosting Trees',GradientBoostingClassifier(n_estimators=200,max_depth=30)),
          ('Random Forest',RandomForestClassifier(n_estimators=200,max_depth=12)),
          ('RGF',RGFClassifier(max_leaf=4000,
                      algorithm="RGF_Sib",
                      test_interval=100,
                      verbose=False)),
          ("XGBoost",XGBClassifier(n_estimators=1000,max_depth=9,learning_rate=.2)),
          ("ERT",ExtraTreesClassifier(n_estimators=1000,max_depth=12))
          ]
  model_results={k[0]:[] for k in models}

  for train_index, test_index in tqdm(skf.split(X, y)):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      scl=StandardScaler()
      X_train=scl.fit_transform(X_train)
      X_test=scl.transform(X_test)
      for m in models:
        m[1].fit(X_train,y_train)
        model_results[m[0]].append(roc_auc_score(y_test,m[1].predict_proba(X_test)[:,1]))
  for label,res in model_results.items():
    print(label,' AUC: ',np.mean(res))