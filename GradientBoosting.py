from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)
gbrt=GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)
Accuracy_train=gbrt.score(X_train,y_train)
Accuracy_test=gbrt.score(X_test,y_test)
print('Train Accuracy RF - ',Accuracy_train)
print('Test Accuracy RF - ',Accuracy_test)
import matplotlib.pyplot as plt
plt.plot(gbrt.feature_importances_,'o')
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
    
