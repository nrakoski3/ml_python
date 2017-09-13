from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


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


# run main() if file is called
if __name__ == "__main__":
    main()