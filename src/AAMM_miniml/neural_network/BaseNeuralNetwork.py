from abc import ABC, abstractmethod
import numpy as np

class BaseNeuralNetwork(ABC):
    def __init__(self, input_size=10, hidden_size=10, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, y_pred):
        # Compute gradients
        d_loss_a2 = -(y_true - y_pred)
        d_a2_z2 = self.sigmoid_derivative(y_pred)
        d_z2_W2 = self.a1
        
        d_loss_W2 = np.dot(d_z2_W2.T, d_loss_a2 * d_a2_z2)
        d_loss_b2 = np.sum(d_loss_a2 * d_a2_z2, axis=0, keepdims=True)
        
        d_loss_a1 = np.dot(d_loss_a2 * d_a2_z2, self.W2.T)
        d_a1_z1 = self.sigmoid_derivative(self.a1)
        d_z1_W1 = X
        
        d_loss_W1 = np.dot(d_z1_W1.T, d_loss_a1 * d_a1_z1)
        d_loss_b1 = np.sum(d_loss_a1 * d_a1_z1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.lr * d_loss_W2
        self.b2 -= self.lr * d_loss_b2
        self.W1 -= self.lr * d_loss_W1
        self.b1 -= self.lr * d_loss_b1
    
    def fit(self, X, y, epochs=1000):
        self.input_size = X.shape[1]
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        return self.forward(X)