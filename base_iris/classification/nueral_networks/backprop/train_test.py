from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import time


def train(data_namedtuple, layer=10, epochs=200):
    # Data
    X_train = data_namedtuple.X_train
    X_test = data_namedtuple.X_test
    y_train = data_namedtuple.y_train
    y_test = data_namedtuple.y_test

    # Param
    param = {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,'learning_rate_init': 0.2}

    # Train
    start = time.clock()
    clf = MLPClassifier(hidden_layer_sizes=layer, max_iter=epochs, **param)
    clf.fit(X_train, y_train)
    train_time = (time.clock() - start)

    # Test
    # clf.predict()
    # clf.predict_proba()  # Cross-Entropy Loss
    # Cross_Val
    #scores = cross_val_score(clf, X_test, y_test, cv=5)
    scores = clf.score(X_test, y_test)

    #return scores.mean(), scores.std()*2, train_time
    return scores, train_time


# Hyper Params
###############
# MLPClassifier(
# activation='relu',
# alpha=1e-05,
# batch_size='auto',
# beta_1=0.9,
# beta_2=0.999,
# early_stopping=False,
# epsilon=1e-08
# hidden_layer_sizes            # Number Hidden Layers
# learning_rate='constant',
# learning_rate_init=0.001,
# max_iter=200,                 # Epochs
# momentum=0.9,
# nesterovs_momentum=True
# power_t=0.5, random_state=1
# shuffle=True,
# solver='lbfgs'
# tol=0.0001,
# validation_fraction=0.1,
# verbose=False,
# warm_start=False

# run main() if file is called
# if __name__ == "__main__":
#    main()
