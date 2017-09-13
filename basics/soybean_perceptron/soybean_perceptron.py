import matplotlib.pyplot as plt
import numpy as np

import basics.datasets.import_df as im
import basics.tools.perceptron as per
import basics.tools.plot_tools as tool


def main():
    # import data into data frame
    df = im.soybean_training()
    # EXPERIMENT,YEAR,LOCATION,VARIETY,FAMILY,CHECK,RM,REPNO,YIELD,CLASS_OF
    # 09YT000052  2009  3310  V000016  FAM05619  True  3.9  1  62.548835  0
    print(df.head())

    # select first 100 locations
    y = df.iloc[0:100, 5].values
    print(y)
    # get column datatypes
    # dtypeCount = [df.iloc[:, i].apply(type).value_counts() for i in range(df.shape[1])]
    # print(dtypeCount)
    y = np.where(y, 1, -1)

    # first 100 locations and yeilds
    X = df.iloc[0:100, [6, 8]].values

    print(X)

    # plot data using matplotlib
    #dftool.iris_versicolor_setosa(X)

    # *******************************************************************

    # train perceptron
    ppn = per.Perceptron(eta=0.1, n_iter=10)

    ppn.fit(X, y)

    # plot
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')

    plt.tight_layout()
    plt.savefig('./perceptron_soybean_train.png', dpi=300)
    plt.show()

    # *******************************************************************

    # Plot Decision Regions
    tool.plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('./perceptron_soybean_decRegions.png', dpi=300)
    plt.show()

# run main() if file is called
if __name__ == "__main__":
    main()