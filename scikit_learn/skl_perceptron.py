from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import scikit_learn.tools.plot_tools as pt


def main():
    # Data
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))

    # Split data into 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Standardize Data
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # *******************************************************************

    # Train Perceptron
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)

    # Outputs
    # y_test.shape
    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # Decision Regions
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    pt.plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=ppn, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
    plt.show()

    # *******************************************************************

    # Sigmoid
    z = np.arange(-7, 7, 0.1)
    phi_z = pt.sigmoid(z)

    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')

    # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)

    plt.tight_layout()
    # plt.savefig('./figures/sigmoid.png', dpi=300)
    plt.show()

    # *******************************************************************

    # Cost 1 & Cost 0
    z = np.arange(-10, 10, 0.1)
    phi_z = pt.sigmoid(z)

    c1 = [pt.cost_1(x) for x in z]
    plt.plot(phi_z, c1, label='J(w) if y=1')

    c0 = [pt.cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel('$\phi$(z)')
    plt.ylabel('J(w)')
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('./figures/log_cost.png', dpi=300)
    plt.show()


# run main() if file is called
if __name__ == "__main__":
    main()
