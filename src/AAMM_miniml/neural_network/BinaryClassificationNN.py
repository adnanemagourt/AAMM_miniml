import numpy as np
from .BaseNeuralNetwork import BaseNeuralNetwork

class BinaryClassificationNN(BaseNeuralNetwork):
    def compute_loss(self, y_true, y_pred):
        # Binary cross-entropy loss
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # To avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y_true, y_pred):
        # # Compute gradients for binary classification
        # d_loss_a2 = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        # d_a2_z2 = self.sigmoid_derivative(y_pred)
        # d_z2_W2 = self.a1
        
        # d_loss_W2 = np.dot(d_z2_W2.T, d_loss_a2 * d_a2_z2)
        # d_loss_b2 = np.sum(d_loss_a2 * d_a2_z2, axis=0, keepdims=True)
        
        # d_loss_a1 = np.dot(d_loss_a2 * d_a2_z2, self.W2.T)
        # d_a1_z1 = self.sigmoid_derivative(self.a1)
        # d_z1_W1 = X
        
        # d_loss_W1 = np.dot(d_z1_W1.T, d_loss_a1 * d_a1_z1)
        # d_loss_b1 = np.sum(d_loss_a1 * d_a1_z1, axis=0, keepdims=True)

        d_loss_a2 = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        d_a2_z2 = self.sigmoid_derivative(y_pred)
        d_z2_W2 = self.a1

        # Ensure element-wise multiplication results in correct shape
        d_loss_W2 = np.dot(d_z2_W2.T, d_loss_a2 * d_a2_z2)
        d_loss_b2 = np.sum(d_loss_a2 * d_a2_z2, axis=0, keepdims=True)

        d_loss_a1 = np.dot(d_loss_a2 * d_a2_z2, self.W2.T)  # This line should work if shapes are correct
        d_a1_z1 = self.sigmoid_derivative(self.a1)
        d_z1_W1 = X

        d_loss_W1 = np.dot(d_z1_W1.T, d_loss_a1 * d_a1_z1)
        d_loss_b1 = np.sum(d_loss_a1 * d_a1_z1, axis=0, keepdims=True)

        
        # Update weights and biases
        self.W2 -= self.lr * d_loss_W2
        self.b2 -= self.lr * d_loss_b2
        self.W1 -= self.lr * d_loss_W1
        self.b1 -= self.lr * d_loss_b1

