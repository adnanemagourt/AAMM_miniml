
# Import the necessary libraries
import numpy as np



# Define the Perceptron class
class SLP:
    """A Single-Layer Perceptron (SLP) binary classifier.
    
    Parameters
    ----------
    hidden_layer_size : int, default=20
        The number of neurons in the hidden layer.
    max_iter : int, default=100
        The maximum number of iterations.
    learning_rate : float, default=237
        The learning rate.
    
    Attributes
    ----------
    nu : float
        The learning rate.
    n : int
        The number of neurons in the hidden layer.
    p : int
        The number of features.
    epochs : int    
        The maximum number of iterations.
    weights_hidden : ndarray
        The weights of the hidden layer.
    bias_hidden : ndarray
        The bias of the hidden layer.
    weights_output : ndarray
        The weights of the output layer.
    bias_output : float
        The bias of the output layer.
    ListOfMSE : list
        The list of Mean Squared Errors (MSE) for each iteration.
        
    """
    def __init__(self, hidden_layer_size = 20, max_iter = 100, learning_rate = 237):
        self.nu = learning_rate
        self.n = hidden_layer_size
        self.p = 0
        self.epochs = max_iter



        self.ListOfMSE = []

        

    def activation(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid activation function
    


    def getCost(self, X, y, weights_hidden, bias_hidden, weights_output, bias_output):
        hidden_layer_input = np.dot(X, weights_hidden.T) + bias_hidden
        hidden_layer_output = self.activation(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
        output_layer_output = self.activation(output_layer_input)
        return np.mean((y - output_layer_output) ** 2) / 2
        
    def OneIterationOutputBias(self, X, y, weights_hidden, bias_hidden, weights_output, bias_output ):
        dx = 0.001
        a = self.getCost(X, y, weights_hidden, bias_hidden, weights_output, bias_output)
        b = self.getCost(X, y, weights_hidden, bias_hidden, weights_output, bias_output + dx)

        # now update the bias_output
        self.bias_output -= self.nu * (b - a) / dx
    
    def oneIterationOutputWeights(self, X, y, weights_hidden, bias_hidden, weights_output, bias_output):
        dx = 0.001

        for i in range(self.n):
            v = np.zeros(self.n)
            v[i] = dx
            a = self.getCost(X, y, weights_hidden, bias_hidden, weights_output + v, bias_output)
            b = self.getCost(X, y, weights_hidden, bias_hidden, weights_output, bias_output)
            self.weights_output[i] -= self.nu * (a - b) / dx

    def OneIterationHiddenBias(self, X, y, weights_hidden, bias_hidden, weights_output, bias_output):
        dx = 0.001

        for i in range(self.n):
            v = np.zeros(self.n)
            v[i] = dx
            a = self.getCost(X, y, weights_hidden, bias_hidden + v, weights_output, bias_output)
            b = self.getCost(X, y, weights_hidden, bias_hidden, weights_output, bias_output)
            self.bias_hidden[i] -= self.nu * (a - b) / dx

    def OneIterationHiddenWeights(self, X, y, weights_hidden, bias_hidden, weights_output, bias_output):
        dx = 0.001

        for i in range(self.n):
            for j in range(self.p):

                # v is a matrix of zeros
                v = np.zeros((self.n, self.p))
                v[i][j] = dx


                a = self.getCost(X, y, weights_hidden + v, bias_hidden, weights_output, bias_output)
                b = self.getCost(X, y, weights_hidden, bias_hidden, weights_output, bias_output)
                self.weights_hidden[i][j] -= self.nu * (a - b) / dx


    def fit(self, X, y):
        self.p = X.shape[1]



        

        # we will set the seed to always get the same results
        np.random.seed(42)

        self.weights_hidden = np.random.randn(self.n , self.p)
        self.bias_hidden = np.random.randn(self.n)
        self.weights_output = np.random.randn(self.n)
        self.bias_output = np.random.randn(1)

        self.ListOfMSE.append(self.getCost(X, y, self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output))


        for i in range(self.epochs):
            self.OneIterationOutputBias(X, y, self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output)
            self.oneIterationOutputWeights(X, y, self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output)
            self.OneIterationHiddenBias(X, y, self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output)
            self.OneIterationHiddenWeights(X, y, self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output)

            # update the best parameters
            a = self.getCost(X, y, self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output)
            self.ListOfMSE.append(a)

            
            
            # print(f"Epoch {i + 1}, Cost: {self.getCost(X, y, self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output)}")

    

    
    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_hidden.T) + self.bias_hidden
        hidden_layer_output = self.activation(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_output) + self.bias_output
        output_layer_output = self.activation(output_layer_input)
        return np.round(output_layer_output).astype(int)

    def getErrors(self):
        return self.ListOfMSE



if __name__ == "__main__":
    
    # Generate some data with 1000 samples and 2 features
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)

    # Create an instance of the SLP class
    slp = SLP( hidden_layer_size = 30, max_iter = 50, learning_rate = 70)

    # Train the model
    slp.fit(X, y)

    # Make predictions
    y_pred = slp.predict(X)

    l = slp.getErrors()
    print(l)

    # Print the accuracy
    print(f"Accuracy: {np.mean(y == y_pred)}")

# Output:

