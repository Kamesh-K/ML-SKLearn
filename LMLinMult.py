import sklearn
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
X,y=make_blobs(random_state=42)
plt.scatter(X[:,0],X[:,1],c=y,s=60,cmap=mglearn.cm3)
plt.show()
#Linear SVM
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
lsvm=LinearSVC().fit(X,y)
print('Test Accuarcy - ',lsvm.score(X_test,y_test))

