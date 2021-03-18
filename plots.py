import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree

def plot_clf(clf, title, x_train, y_train):
    x = x_train[:, :2]
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, .02), np.arange(y_min, y_max, .02))

    clf.fit(x, y_train)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.figure(figsize=[10,10])
    plt.contourf(xx, yy, z, cmap=plt.cm.RdYlGn, alpha=0.8)
    plt.scatter(x[:,0], x[:,1], c=y_train, cmap=plt.cm.RdYlGn, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('n° of Collision Avoidance service instances')
    plt.ylabel('n° of See Through service instances')
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()

def my_plot_tree(clf_tree):
    features_list = ['n_instances_CA', 'n_instances_ST', 'n_instances_VS']
    label_list = ['0', '1', '2', '3']
    plt.figure()
    plot_tree(clf_tree, class_names=label_list, feature_names=features_list, filled=True)
    plt.show()