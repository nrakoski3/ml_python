import matplotlib.pyplot as plt
import numpy as np
import basics.datasets.import_df as im
import basics.tools.stochastic_adaline as sa
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
    # Standardize Features and Train

    # standardize
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # *******************************************************************

    # init Train
    ada = sa.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)

    tool.plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('./iris_stochadaline_train', dpi=300)
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')

    plt.tight_layout()
    plt.savefig('./iris_stochadaline_cost.png', dpi=300)
    plt.show()

# run main() if file is called
if __name__ == "__main__":
    main()