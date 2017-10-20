from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras import losses


def main():
    model = Sequential()
    model.add(Dense(output_dim=7, input_dim=3))
    model.add(Activation("sigmoid"))
    model.add(Dense(output_dim=1))
    model.add(Activation("sigmoid"))

    print(losses)


    return

if __name__ == "__main__":
    main()
