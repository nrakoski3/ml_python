import matplotlib.pyplot as plt
import numpy as np
import basics.datasets.import_df as im
import basics.tools.adaline as a
import basics.datasets.plot_df as plot
import basics.tools.plot_tools as tool


def main():
    # import data into data frame
    df = im.iris()

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # plot data using matplotlib
    #plot.iris_versicolor_setosa(X)

    # *******************************************************************

    # Train Adaline, 2 different learning rates, plot results
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ada1 = a.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = a.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.tight_layout()
    plt.savefig('./iris_adaline_initTrain.png', dpi=300)
    plt.show()

    # *******************************************************************
    # Standardize Features and Retrain

    # standardize
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # retrain
    ada = a.AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    tool.plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./iris_adaline_standGD_dec.png', dpi=300)
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')

    plt.tight_layout()
    plt.savefig('./iris_adaline_reTrain.png', dpi=300)
    plt.show()

# run main() if file is called
if __name__ == "__main__":
    main()
