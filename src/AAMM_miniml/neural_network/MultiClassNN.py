from .BaseNeuralNetwork import BaseNeuralNetwork
import numpy as np


class MultiClassNN(BaseNeuralNetwork):
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability improvement
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        # Categorical cross-entropy loss
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # To avoid log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, X, y_true, y_pred):
        # Compute gradients for multi-class classification
        d_loss_a2 = y_pred - y_true
        d_z2_W2 = self.a1
        
        d_loss_W2 = np.dot(d_z2_W2.T, d_loss_a2)
        d_loss_b2 = np.sum(d_loss_a2, axis=0, keepdims=True)
        
        d_loss_a1 = np.dot(d_loss_a2, self.W2.T)
        d_a1_z1 = self.sigmoid_derivative(self.a1)
        d_z1_W1 = X
        
        d_loss_W1 = np.dot(d_z1_W1.T, d_loss_a1 * d_a1_z1)
        d_loss_b1 = np.sum(d_loss_a1 * d_a1_z1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.lr * d_loss_W2
        self.b2 -= self.lr * d_loss_b2
        self.W1 -= self.lr * d_loss_W1
        self.b1 -= self.lr * d_loss_b1

