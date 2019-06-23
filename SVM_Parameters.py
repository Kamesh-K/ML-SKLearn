import sklearn
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
X,y=mglearn.tools.make_handcrafted_dataset()
svm=SVC(kernel='rbf',C=10,gamma=0.1).fit(X,y)

plt.scatter(X[:,0],X[:,1],s=60,c=y,cmap=mglearn.cm2)
sv = svm.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none', zorder=10, linewidth=3)
plt.show()

plt.scatter(X[:, 0], X[:, 1], s=60, c=y, cmap=mglearn.cm2)
# plot support vectors
sv = svm.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',edgecolors='black', zorder=5, linewidth=2)
plt.show()
