import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets


def main():

    # Get data from UCI database
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))

    # Get data from CSV as Pandas Dataframe. Set column names so they're easy to call, but not in the dataframe
    df = pd.read_csv('/Users/nrakoski/Documents/GitHub/ml_python/'
                     'soybean/data/TRAINING_DATASET.csv')  # header=None, skiprows=1)
    cols = df.columns  # get attribute names as strings
    df = pd.read_csv('/Users/nrakoski/Documents/GitHub/ml_python/'
                     'soybean/data/TRAINING_DATASET.csv', header=None, skiprows=1)
    df.columns = cols

    # Replace Unknown Values
    df = df.replace('?', np.nan)

    # Separate Target and Attributes(data)
    target = df.iloc[:, 5].values
    target = np.where(target, 1, -1)  # Normalize Target Data to binary -1 or 1
    data1 = df.iloc[:, 0:4]
    data2 = df.iloc[:, 6:-1]
    data = pd.concat([data1, data2], axis=1)

    # If any Categories, encode them
    data_catdf = data.select_dtypes(include=[object])
    if data_catdf.shape[1] != 0:
        print("DF of Category values")
        print(data_catdf.shape)
        print(data_catdf.columns)
        print(data_catdf.head(5))
        le = LabelEncoder()
        data_catdf = data_catdf.apply(le.fit_transform)
        print("DF of Label encoded Category values")
        print(data_catdf.shape)
        print(data_catdf.columns)
        print(data_catdf.head(5))

        # All Continuous
        data_contdf = data.select_dtypes(exclude=[object])
        print("DF of Continuous values")
        print(data_contdf.shape)
        print(data_contdf.columns)
        print(data_contdf.head(5))

    # Merge Categorial and Continuous, print dataframe infor for inspection
    df = pd.concat([data_catdf, data_contdf], axis=1)
    print(df.dtypes)
    print("DF of all data")
    print(df.shape)
    print(df.columns)
    print(df.head(5))
    print(df.tail(5))

    # Save to new CSV for java use
    pd.DataFrame.to_csv(path_or_buf='/Users/nrakoski/Documents/GitHub/ml_python/soybean/data/encoded_TRAINING_DATASET.csv')


    # Transfer to Numpy Matrix for scikit learn use
    data = df.values
    print(data.shape)
    print(target.shape)

    # Split Data into Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

    return

if __name__ == "__main__":
    main()