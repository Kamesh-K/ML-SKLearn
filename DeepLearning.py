from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import mglearn
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7)
#mlp = MLPClassifier(algorithm='l-bfgs', random_state=0).fit(X_train, y_train)
mlp=MLPClassifier(solver='lbfgs',random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=60, cmap=mglearn.cm2)
print(mlp.score(X_train,y_train))
print(mlp.score(X_test,y_test))
plt.show()

