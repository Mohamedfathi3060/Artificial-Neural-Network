import numpy as np
from NN import X_train, X_test, y_train, y_test
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class Sigmoid:
    def behavior(self, x):
        return 1 / (1 + np.exp(-x))

    def Activate(self, inp):
        func = np.vectorize(self.behavior)
        return func(inp)


class Linear:
    def behavior(self, x):
        return x

    def Activate(self, inp):
        func = np.vectorize(self.behavior)
        return func(inp)


class Layer:
    def __init__(self, N, ActivFunc):
        self.Num_Neuron = N
        self.prevLayer = None
        self.nextLayer = None
        self.ActivationFunction = ActivFunc
        self.weights = None
        self.Out = None
        self.In = None

    def compile(self):
        # self.weights = np.zeros((self.Num_Neuron, self.prevLayer.Num_Neuron + 1))
        # self.weights[:, -1] = 1
        self.weights = np.random.rand(self.Num_Neuron, self.prevLayer.Num_Neuron + 1)

    def run(self, In):
        self.In = np.vstack((In, np.array([[1]])))
        sop = np.dot(self.weights, self.In)
        self.Out = self.ActivationFunction.Activate(sop)
        return self.Out


class ANN:
    def __init__(self,A, epoch):
        self.layers = []
        self.Alpha = A
        self.epoch = epoch

    def setLayers(self, layers):
        self.layers = layers

    def addLayer(self, layer):
        self.layers.append(layer)

    def compile(self):
        prev = self.layers[0]
        for layer in self.layers[1:]:
            layer.prevLayer = prev
            layer.compile()
            prev = layer

    def fit(self, xTrain, yTrain):
        inputLayer = Layer(xTrain.shape[1], None)
        self.layers[0].prevLayer = inputLayer
        self.layers[0].compile()

        for k in range(self.epoch):
            for i in range(xTrain.shape[0]):
                trans = xTrain[i].reshape(-1, 1)
                for j in range(len(self.layers)):
                    trans = self.layers[j].run(trans)
                self.updateWeights(yTrain[i])

    def updateWeights(self, yTrain):
        outLayer = self.layers[-1]
        # S = outLayer.Out[0] * (1 - outLayer.Out[0]) * (yTrain - outLayer.Out[0])
        S = (yTrain - outLayer.Out[0])
        S = S[0]
        for j in range(outLayer.prevLayer.Num_Neuron + 1):
            outLayer.weights[0][j] = outLayer.weights[0][j] + (self.Alpha * S * outLayer.In[j][0])

        #### HIDDEN LAYERS ######

        for hiddenLayer in self.layers[:-1]:
            for i in range(hiddenLayer.Num_Neuron):
                HiddenError = hiddenLayer.Out[i] * (1 - hiddenLayer.Out[i]) * (S * outLayer.weights[0][i])
                HiddenError = HiddenError[0]
                for j in range(hiddenLayer.prevLayer.Num_Neuron + 1):
                    hiddenLayer.weights[i][j] = hiddenLayer.weights[i][j] + (self.Alpha * HiddenError * hiddenLayer.In[j][0])

    def predict(self, xTest):
        predicted = np.zeros((xTest.shape[0], 1))
        for i in range(xTest.shape[0]):
            trans = xTest[i].reshape(-1, 1)
            for j in range(len(self.layers)):
                trans = self.layers[j].run(trans)
            predicted[i] = trans[0]
        return predicted


layer1 = Layer(8, Sigmoid())
layerOut = Layer(1,Linear())


obj = ANN(0.1, 1000)
obj.setLayers([layer1, layerOut])
obj.compile()
obj.fit(X_train, y_train)

y_pred = obj.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)




# obj.predict(X_test)


# arr = np.array([
#     [0.1, 0.3]
# ])
#
# layer1 = Layer(2, Sigmoid())
# layerOut = Layer(1,Sigmoid())
#
#
# #
# # layer1.weights = np.array([
# #             [0.5, 0.1, 0.4],
# #             [0.62, 0.2, -0.1]
# #         ])
# # layerOut.weights = np.array([
# #     [-0.2, 0.3, 1.83]
# # ])
# #
#
# obj = ANN()
# obj.setLayers([layer1, layerOut])
# obj.compile()
# obj.fit(arr, [])
#


# layer1.compile()
# layerOut.compile()
#
# print(layerOut.run(layer1.run(arr.T)))

