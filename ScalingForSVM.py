import sklearn
from sklearn.svm import SVC
svc=SVC(C=2000,gamma=0.001)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test= train_test_split(cancer.data,cancer.target,random_state=0)
svc.fit(X_train,y_train)
print(svc.score(X_train,y_train))
print(svc.score(X_test,y_test))
min_x_train=X_train.min(axis=0)
max_x_train=X_train.max(axis=0)
range=max_x_train-min_x_train
X_train_scaled=X_train-min_x_train
X_train_scaled=X_train_scaled/range
svc.fit(X_train_scaled,y_train)
print(svc.score(X_train_scaled,y_train))
X_test_scaled=(X_test-min_x_train)/range
print(svc.score(X_test_scaled,y_test))

