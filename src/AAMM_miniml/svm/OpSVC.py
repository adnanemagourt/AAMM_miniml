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
    """
    def __init__(self, C=1, tol=1e-3, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.classifiers = None
        self.classes = None

    def _compute_error(self, i, X, y, alpha, b):
        return (np.dot(alpha * y, X @ X[i, :]) + b) - y[i]

    def _select_second_index(self, i, m, X, y, alpha, b):
        max_step = 0
        selected_j = -1
        E_i = self._compute_error(i, X, y, alpha, b)
        
        # Loop through all possible indices
        for j in range(m):
            if j == i:
                continue
            
            E_j = self._compute_error(j, X, y, alpha, b)
            step = abs(E_i - E_j)
            
            if step > max_step:
                max_step = step
                selected_j = j
        
        return selected_j if selected_j != -1 else (i + 1) % m


    def _fit(self, X, y):
        m, n = X.shape
        alpha = np.zeros(m)
        b = 0
        passes = 0
        
        while passes < self.max_passes:
            num_changed_alphas = 0
            
            for i in range(m):
                Ei = self._compute_error(i, X, y, alpha, b)
                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or (y[i] * Ei > self.tol and alpha[i] > 0):
                    j = self._select_second_index(i, m, X, y, alpha, b)
                    Ej = self._compute_error(j, X, y, alpha, b)
                    
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                    
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * X[i, :] @ X[j, :].T - X[i, :] @ X[i, :].T - X[j, :] @ X[j, :].T
                    if eta >= 0:
                        continue
                    
                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)
                    
                    if abs(alpha[j] - alpha_j_old) < self.tol:
                        continue
                    
                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                    b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * X[i, :] @ X[i, :].T - y[j] * (alpha[j] - alpha_j_old) * X[i, :] @ X[j, :].T
                    b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * X[i, :] @ X[j, :].T - y[j] * (alpha[j] - alpha_j_old) * X[j, :] @ X[j, :].T
                    
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        w = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)
        return w, b




    def _predict(X, w, b):
        return np.sign(X @ w + b)




    def fit(self,X, y):
        self.classes = np.unique(y)
        self.classifiers = []
        
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            w, b = self._fit(X, y_binary)
            self.classifiers.append((w, b))
        
        


    def predict(self, X):
        decision_values = np.zeros((X.shape[0], len(self.classes)))
        
        for i, (w, b) in enumerate(self.classifiers):
            decision_values[:, i] = X @ w + b
        
        return self.classes[np.argmax(decision_values, axis=1)]


    def score(self, X, y):
        return np.mean(self.predict(X) == y)

