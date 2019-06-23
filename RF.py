from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=5)
t1=time.time()
rf=RandomForestClassifier(n_estimators=100,random_state=5,n_jobs=1)
rf.fit(X_train,y_train)
Accuracy_train=rf.score(X_train,y_train)
Accuracy_test=rf.score(X_test,y_test)
t2=time.time()
print('Train Accuracy RF - ',Accuracy_train)
print('Test Accuracy RF - ',Accuracy_test)
plt.plot(rf.feature_importances_, 'o')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90);
plt.show()
print(t2-t1)


