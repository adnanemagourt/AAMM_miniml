import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Classifier

class SVC(Classifier):
    """
    A Support Vector Classifier.
    
    Parameters
    ----------
    C : float, default=1.0
        The regularization parameter.
    kernel : str, default='rbf'
        The kernel to be used. It must be one of 'linear', 'poly', 'rbf', 'sigmoid'.
    degree : int, default=3
        The degree of the polynomial kernel. Ignored by other kernels.
    gamma : float or {'scale', 'auto'}, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - if 'scale', then it uses 1 / (n_features * X.var()) as value of gamma.
        - if 'auto', then it uses 1 / n_features.
    coef0 : float, default=0.0
        Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_iter : int, default=1000
        The maximum number of iterations.
    
    Attributes
    ----------
    C : float
        The regularization parameter.
    kernel : str
        The kernel to be used.
    degree : int
        The degree of the polynomial kernel.
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    coef0 : float
        Independent term in kernel function.
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        The maximum number of iterations.
    alphas : ndarray
        The Lagrange multipliers.
    b : float
        The bias.
    support_ : ndarray
        The support vectors indices.
    support_vectors_ : ndarray
        The support vectors.
    support_labels_ : ndarray
        The labels of the support vectors.
    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
    
    def _kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(X, Y.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                self.gamma = 1 / (X.shape[1] * X.var())
            elif self.gamma == 'auto':
                self.gamma = 1 / X.shape[1]
            K = np.zeros((X.shape[0], Y.shape[0]))
            for i, x in enumerate(X):
                K[i] = np.exp(-self.gamma * np.sum((x - Y) ** 2, axis=1))
            return K
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(X, Y.T) + self.coef0)
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = self._kernel_function(X, X)
        
        # Initialize Lagrange multipliers (alphas) and bias (b)
        alphas = np.zeros(n_samples)
        b = 0
        iter = 0

        while iter < self.max_iter:
            alpha_prev = np.copy(alphas)
            for i in range(n_samples):
                # Calculate error for sample i
                E_i = self._decision_function(X[i], X, y, alphas, b) - y[i]

                if (y[i] * E_i < -self.tol and alphas[i] < self.C) or (y[i] * E_i > self.tol and alphas[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Calculate error for sample j
                    E_j = self._decision_function(X[j], X, y, alphas, b) - y[j]
                    
                    # Save old alphas
                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                    # Compute bounds for alpha[j]
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue

                    # Compute eta (the second derivative of the objective function)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha[j]
                    alphas[j] -= y[j] * (E_i - E_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    if abs(alphas[j] - alpha_j_old) < self.tol:
                        continue

                    # Update alpha[i]
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                    # Compute b1 and b2
                    b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    # Update b
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

            iter += 1
            diff = np.linalg.norm(alphas - alpha_prev)
            if diff < self.tol:
                break

        self.alphas = alphas
        self.b = b
        self.support_ = alphas > 0
        self.support_vectors_ = X[self.support_]
        self.support_labels_ = y[self.support_]
    
    def _decision_function(self, X, support_vectors, support_labels, alphas, b):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        K = self._kernel_function(support_vectors, X)
        return np.sum(alphas[:, np.newaxis] * support_labels[:, np.newaxis] * K, axis=0) + b

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        decision_values = self._decision_function(X, self.support_vectors_, self.support_labels_, self.alphas[self.support_], self.b)
        return np.sign(decision_values)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)







# Example usage
if __name__ == '__main__':
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, -1, -1])

    clf = SVC(C=1.0, kernel='rbf', gamma='auto')
    clf.fit(X, y)
    pred = clf.predict(np.array([[-0.8, -1]]))
    print(pred)  # Output: [1]






