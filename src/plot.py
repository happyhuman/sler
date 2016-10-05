import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


def plot_decision_region(model, df, x_feature, y_feature, target):
    """
    Plot a decision boundary for a classification model with two featues
    :param model: the classification model
    :param df: the pandas DataFrame
    :param x_feature: name of the feature for the X axis in the DataFrame
    :param y_feature: name of the feature for the Y axis in the DataFrame
    :param target: name of the target in the DataFrame
    """
    xs = df[x_feature]
    ys = df[y_feature]
    h = .02
    x_min, x_max = int(xs.min()) - 1, int(xs.max()) + 1
    y_min, y_max = int(ys.min()) - 1, int(ys.max()) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(xs, ys, c=df[target], cmap=plt.cm.Paired)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(range(x_min, x_max, (x_max - x_min) / 5))
    plt.yticks(range(y_min, y_max, (y_max - y_min) / 5))
    plt.title("Decision Boundary")
