import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss Function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Neural Network class
class NeuralNetworkBP:
    """
    A simple feedforward neural network with a single hidden layer.
    
    Parameters
    ----------
    input_size : int
        The number of input neurons.
    hidden_size : int
        The number of hidden neurons.
    output_size : int
        The number of output neurons.
        
    Attributes
    ----------
    input_size : int
        The number of input neurons.
    hidden_size : int
        The number of hidden neurons.
    output_size : int
        The number of output neurons.
    weights_input_hidden : ndarray 
        The weights between the input and hidden layer.
    weights_hidden_output : ndarray
        The weights between the hidden and output layer
    bias_hidden : ndarray
        The bias of the hidden layer.
    bias_output : ndarray
        The bias of the output layer.
    hidden_input : ndarray
        The input to the hidden layer.
    hidden_output : ndarray
        The output of the hidden layer.
    final_input : ndarray
        The input to the output layer.
    final_output : ndarray
        The output of the output layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        np.random.seed(0)

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y, learning_rate):
        # Calculate the error
        loss = mse_loss(y, self.final_output)
        
        # Calculate gradients
        output_error = self.final_output - y
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output -= self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden -= X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        
        return loss

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.backward(X, y, learning_rate)
            # if epoch % 100 == 0:
            #     print(f'Epoch {epoch}, Loss: {loss}')

# Example usage
if __name__ == "__main__":
    # XOR problem
    X = np.array([[0, 0, 4], [0, 1, 5], [1, 0, 3], [1, 1, 0]])
    y = np.array([[0.4], [0.9], [0.1], [0]])

    # generate random data

    np.random.seed(0)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 1, (100, 1))
    
    # Initialize the Neural Network
    nn = NeuralNetworkBP(input_size=10, hidden_size=400, output_size=1)
    
    # Train the Neural Network
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    # Test the Neural Network
    output = nn.forward(X)


    print(max(output))
    print(min(output))



