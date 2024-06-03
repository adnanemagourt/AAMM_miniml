from .BaseNeuralNetwork import BaseNeuralNetwork
import numpy as np


class RegressionNN(BaseNeuralNetwork):
    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for regression
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        # Mean squared error loss
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, y_pred):
        # Compute gradients for regression
        d_loss_a2 = -(y_true - y_pred)
        d_a2_z2 = 1  # Derivative of linear activation is 1
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

