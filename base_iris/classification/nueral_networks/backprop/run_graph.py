from train_test import train
from collections import namedtuple
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def main():
    # Test Varying Training Data Size
    pertrainEx = numpy.arange(1, 10, 1) * 0.1
    print(pertrainEx)
    meanArr, sizeArr, timeArr = percent_train_examples(pertrainEx)
    print(sizeArr)
    print(meanArr)
    # print(timeArr)

    return


def percent_train_examples(perArr):
    # Declare NamedTuple
    Data = namedtuple('Data', 'X_train X_test y_train y_test col_names')
    # Separate Testing Data
    X, X_test, y, y_test, col_names = sep_test_train()

    # Train with portions of Training Data
    sizeArr = numpy.array([])
    meanArr = numpy.array([])
    timeArr = numpy.array([])

    for per in perArr:
        # Grab Portion
        X_train, y_train = train_data_split(X, y, train_size=per)
        size = len(y_train)
        print(size)
        scale = Data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, col_names=col_names)

        # Train & Test
        mean, time = train(scale)

        meanArr = numpy.append(meanArr, [mean])
        sizeArr = numpy.append(sizeArr, [size])
        timeArr = numpy.append(timeArr, [time])

    # Use All Data
    print(len(y))
    scale = Data(X_train=X, X_test=X_test, y_train=y, y_test=y_test, col_names=col_names)

    # Train & Test
    mean, time = train(scale)

    meanArr = numpy.append(meanArr, [mean])
    sizeArr = numpy.append(sizeArr, [size])
    timeArr = numpy.append(timeArr, [time])
    # Graph

    return meanArr, sizeArr, timeArr


def sep_test_train():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    print(len(X))
    y = iris.target
    print('Class labels:', numpy.unique(y))

    # Separate Testing Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    print(len(y_train))
    print(len(y_test))

    # Scale Features
    # stdsc = StandardScaler()
    # X_train_std = stdsc.fit_transform(X_train)
    # X_test_std = stdsc.transform(X_test)
    # return X_train_std, X_test_std, y_train, y_test, numpy.unique(y)

    return X_train, X_test, y_train, y_test, numpy.unique(y)


def train_data_split(X_train, y_train, train_size):
    # Grab portion of Training Data
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.0, train_size=train_size, random_state=0)

    return X_train, y_train


def balance_data():

    return


def graph_time_epochs():

    return


def graph_error_numEx():

    return


def graph_error_epochs():

    return


def graph_error_layers():

    return

# run main() if file is called
if __name__ == "__main__":
    main()