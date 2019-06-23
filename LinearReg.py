import sklearn
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X,y=mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
lr=LinearRegression().fit(X_train,y_train)
print("Test Accuracy - %f" %lr.score(X_test,y_test))
print("Train Accuracy - %f" %lr.score(X_train,y_train))
print(lr.intercept_)
