import matplotlib.pyplot as plt
import numpy as np

import basics.datasets.import_df as im
import basics.tools.perceptron as per
import basics.tools.plot_tools as tool


def main():
    # import data into data frame
    df = im.iris()

    print(df.head())

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    print(y)
    y = np.where(y == 'Iris-setosa', -1, 1)
    print(y)

    # extract sepal length and petal length
    # sepal +  + petal +
    # 5.1  3.5  1.4  0.2  Iris-setosa
    X = df.iloc[0:100, [0, 2]].values
    print(X)

    # plot data using matplotlib
    # plot.iris_versicolor_setosa(X)

    # *******************************************************************

    # train perceptron

    #for()
    etan = 0.9
    ppn = per.Perceptron(eta=etan, n_iter=10)

    ppn.fit(X, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # plot
    ax[0].plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Number of misclassifications')
    ax[0].set_title("Perceptron - Learning rate: " + str(etan))

    # plt.tight_layout()
    # plt.savefig('./perceptron_iris_train.png', dpi=300)
    # plt.show()

    # *******************************************************************

    # Plot Decision Regions
    tool.plot_decision_regions(X, y, classifier=ppn)
    ax[1].set_xlabel('sepal length [cm]')
    ax[1].set_ylabel('petal length [cm]')
    ax[1].legend(loc='upper left')

    plt.tight_layout()
    etan = int(etan*10)
    plt.savefig("./train_decRegions_lr" + str(etan) + ".png", dpi=300)
    plt.show()


# run main() if file is called
if __name__ == "__main__":
    main()
