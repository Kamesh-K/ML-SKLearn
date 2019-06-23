import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
scaler=MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=Normalizer()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
from sklearn.svm import SVC

svm=SVC(C=100)
svm.fit(X_train,y_train)
print(svm.score(X_train,y_train))
print(svm.score(X_test,y_test))

X_test_scaled=scaler.transform(X_test)
sv2=svm.fit(X_train_scaled,y_train)
print(svm.score(X_test_scaled,y_test))
