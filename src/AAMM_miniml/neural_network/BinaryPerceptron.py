import numpy as np


class Perceptron:

    def predict(self, X):
        X = self.add_bias(X)
        convert_to_class = lambda c: self.classes[1] if c >= 0 else self.classes[0]
        return np.vectorize(convert_to_class)(X.dot(self.weights))
    
    def fit(self, X, y, epochs=1000, lr=0.001):
        self.classes = np.unique(y)
        X = self.add_bias(X)
        y = (y == self.classes[1]) * 1
        self.weights = np.zeros(X.shape[1])
        for _ in range(epochs): 
            self.weights += lr * np.dot((y - self.predict(X[:,1:])), X)
            
    def add_bias(self,X):
        return np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    



class Perceptron:
    def __init__(self, n_features=3, weights=None, bias=0, 
                 epoch=100, alpha=0.1):
        self.weights = np.zeros(n_features) if weights == None else weights
        self.bias = bias
        self.epoch = epoch
        self.alpha = alpha
        
    #methods
    def weighted_sum(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def activation(self, sum_):
        return (sum_ >= 0).astype(int)
    
    def predict(self, X):
        return self.activation(self.weighted_sum(X))
    
    def compute_error(self, X, y):
        return  y-self.predict(X)
    
    def update_weights(self, error, X):
        for xi, errori in zip(X, error):
            self.weights += errori * xi * self.alpha
            self.bias += errori * self.alpha
            
    def fit(self, X, y):
        for i in range(self.epoch):
            error = self.compute_error(X, y)
            self.update_weights(error, X)
            print(f'{i} error : {np.mean(error)}')
    




########################################################################""

"""
Разработанный Адриелу Ванг от ДанСтат Консульти́рования
"""

import torch
class PerceptronMain:
    def __init__(self, layer_sizes, activation_function, optimizer_function, weight_decay= 0.0, add_bias = True):
        self.layer_sizes = layer_sizes
        self.activation_function = TorchActivations.activation(activation_function)
        self.activation_derivative = TorchActivations.derivative(activation_function)
        self.optimizer_function = optimizer_function
        self.add_bias = add_bias
        self.weight_decay = weight_decay
        #self.optimizer_params = {}
        self.initialize_weights()
        if self.add_bias:
            self.layer_sizes[0] += 1

    def initialize_weights(self, dtype=torch.float64):
        self.weights = [torch.randn(n, m, dtype=dtype) for n, m in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.velocity = None
        self.squared_gradients = [torch.zeros_like(w) for w in self.weights]

    def forward(self, X):
        self.a_values = [X]
        for w in self.weights:
            self.a_values.append(self.activation_function(self.a_values[-1] @ w))
        return self.a_values[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        gradients = [torch.zeros_like(w) for w in self.weights]

        if y.dim() == 1:
            y = y.view(-1, 1)

        delta = (self.a_values[-1] - y) * self.activation_derivative(self.a_values[-2] @ self.weights[-1])
        gradients[-1] = self.a_values[-2].t() @ delta + self.weight_decay * self.weights[-1]

        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].t()) * self.activation_derivative(self.a_values[i] @ self.weights[i])
            gradients[i] = self.a_values[i].t() @ delta + self.weight_decay * self.weights[i]

        return gradients

    def optimize(self, gradients, learning_rate, momentum):
        self.weights, self.velocity = self.optimizer_function(self.weights, gradients, learning_rate, self.weight_decay, momentum = momentum, velocity=self.velocity, squared_gradients=self.squared_gradients)

    def fit(self, X, y, epochs, batch_size, learning_rate, momentum = 0, epoch_step=100):
        step = epoch_step
        current_epochs = epochs
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        self.initialize_weights(dtype=X.dtype)

        if self.add_bias:
            # Add a column of 1s to the input data
            X = torch.cat((X, torch.ones((X.shape[0], 1))), dim=1)
        
        while current_epochs > 0:
            print(f"Trying {current_epochs} epochs.")
            
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    for epoch in range(current_epochs):
                        for i in range(0, X.shape[0], batch_size):
                            X_batch = X[i:min(i + batch_size, X.shape[0])]
                            y_batch = y[i:min(i + batch_size, y.shape[0])]
                            self.forward(X_batch)
                            gradients = self.backward(X_batch, y_batch, learning_rate)
                            
                            self.optimize(gradients = gradients, learning_rate = learning_rate, momentum = momentum)

                        if w:
                            raise RuntimeWarning("Overflow encountered during training.")

                print(f"Training successful with {current_epochs} epochs.")
                break

            except RuntimeWarning:
                print(f"Warning encountered with {current_epochs} epochs. Reducing the number of epochs.")
                current_epochs -= step

    def predict(self, X):
        X = X.to(self.weights[0].dtype)
        if self.add_bias:
            X = torch.cat((X, torch.ones((X.shape[0], 1), dtype=X.dtype)), dim=1)
        for w in self.weights[:-1]:
            X = self.activation_function(X @ w)
        return X @ self.weights[-1]
    
class Optimizers:
    @staticmethod
    def sgd_optimizer(weights, gradients, learning_rate, weight_decay, momentum=0.0, velocity=None, **kwargs):
        if velocity is None:
            velocity = [torch.zeros_like(w) for w in weights]

        # update the velocity
        velocity = [momentum * v + (1 - momentum) * g for v, g in zip(velocity, gradients)]
        
        # update the weights
        new_weights = [w - learning_rate * v for w, v in zip(weights, velocity)]

        return new_weights, velocity

    @staticmethod
    def adagrad_optimizer(weights, gradients, learning_rate, weight_decay, squared_gradients=None, eps=1e-8, **kwargs):
        if squared_gradients is None:
            squared_gradients = [torch.zeros_like(w) for w in weights]

        # update the squared gradients
        new_squared_gradients = [sg + g ** 2 for sg, g in zip(squared_gradients, gradients)]
        
        # update the weights
        new_weights = [w - learning_rate / (torch.sqrt(sg) + eps) * (g + weight_decay * w) for w, sg, g in zip(weights, new_squared_gradients, gradients)]

        return new_weights, new_squared_gradients


class TorchActivations:
    activations = {
        'sigmoid': lambda x: 1 / (1 + torch.exp(-x)),
        'tanh': lambda x: torch.tanh(x),
        'relu': lambda x: torch.max(torch.zeros_like(x), x),
        'relu_squared': lambda x: torch.pow(torch.max(torch.zeros_like(x), x), 2),
        'linear': lambda x: x,
        'softmax': lambda x: torch.exp(x) / torch.sum(torch.exp(x), axis=0),
        'logistic': lambda x: 1 / (1 + torch.exp(-x))  # Logistic is the same as sigmoid
    }
    
    derivatives = {
        'sigmoid': lambda x: TorchActivations.activations['sigmoid'](x) * (1 - TorchActivations.activations['sigmoid'](x)),
        'tanh': lambda x: 1 - torch.pow(TorchActivations.activations['tanh'](x), 2),
        'relu': lambda x: (x > 0).float(),
        'relu_squared': lambda x: 2 * torch.max(torch.zeros_like(x), x),
        'linear': lambda x: torch.ones_like(x),
        'softmax': lambda x: TorchActivations.activations['softmax'](x) * (1 - TorchActivations.activations['softmax'](x)),
        'logistic': lambda x: TorchActivations.activations['logistic'](x) * (1 - TorchActivations.activations['logistic'](x))  # Logistic is the same as sigmoid
    }
    
    @staticmethod
    def activation(activation_name):
        return TorchActivations.activations.get(activation_name, None)
    
    @staticmethod
    def derivative(activation_name):
        return TorchActivations.derivatives.get(activation_name, None)


########################################################################""













