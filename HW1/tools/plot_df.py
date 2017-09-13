import matplotlib.pyplot as plt


def iris_versicolor_setosa(X):

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('./iris_versicolor_setosa.png', dpi=300)

    return plt.show()