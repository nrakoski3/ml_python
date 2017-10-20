from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras import losses
from HW2.flappybird.edited_flappyBirdGenAlg.flappy import main as flappy_main


def main():

    result = flappy_main()
    print(result)

    return

if __name__ == "__main__":
    main()

# param = HyperParam()
# param = change_hyperparam_settings(param)
# def change_hyperparam_settings(param):
#     return param
#
#
# class HyperParam(Params):
#     def __init__(self):
#         # self.name = name
#         # self.lr = 0.01
#         # self.decay = 1e-6
#         # self.momentum = 0.9
#         # self.nesterov = True
#         # self.loss = "mse"
#         # # self.optimizer = sgd
#         # self.metrics = ["accuracy"]
#         self.lr = 0.01
#         self.decay = 1e-6
#         self.momentum = 0.9
#         self.nesterov = True
#         self.loss = "mse"
#         # self.optimizer = sgd
#         self.metrics = ["accuracy"]
#
#     def set_hyperparams(self, lr, decay, momentum, nesterov, loss, metrics):
#         self.lr = lr
#         self.decay = decay
#         self.momentum = momentum
#         self.nesterov = nesterov
#         self.loss = loss
#         # self.optimizer = sgd
#         self.metrics = metrics
#         return self