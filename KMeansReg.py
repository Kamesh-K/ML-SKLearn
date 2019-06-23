import sklearn
import mglearn
from sklearn.neighbors import KNeighborsRegressor
X,y= mglearn.datasets.make_wave(n_samples=40)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)
print(reg.score(X_test,y_test))



