import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)
tree=DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print('Train Accuracy DT  - ',tree.score(X_train,y_train))
print('Test Accuracy RF - ',tree.score(X_test,y_test))
rf=RandomForestClassifier(n_estimators=100,random_state=0)
rf.fit(X_train,y_train)
Accuracy_train=rf.score(X_train,y_train)
Accuracy_test=rf.score(X_test,y_test)
print('Train Accuracy RF - ',Accuracy_train)
print('Test Accuracy RF - ',Accuracy_test)

