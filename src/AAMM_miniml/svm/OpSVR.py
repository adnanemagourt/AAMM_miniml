import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Regressor

class SVR(Regressor):
    """A Support Vector Regressor.
    
    Parameters
    ----------
    C : float, default=1
        The regularization parameter.
    epsilon : float, default=0.1
        The epsilon-tube within which no penalty is associated in the training loss function.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_passes : int, default=5
        The maximum number of passes over the training data.
        
    Attributes
    ----------
    C : float
        The regularization parameter.
    epsilon : float
        The epsilon-tube within which no penalty is associated in the training loss function.
    tol : float
        Tolerance for stopping criterion.
    max_passes : int
        The maximum number of passes over the training data.
    w : ndarray
        The weights.
    b : float
        The bias.
    
    """
    def __init__(self, C=1, epsilon=0.1, tol=1e-3, max_passes=5):
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.max_passes = max_passes
        self.w = None
        self.b = None

    def _compute_error_svr(self, i, X, y, alpha, alpha_star, b, epsilon):
        return (np.dot((alpha - alpha_star), X @ X[i, :].T) + b) - y[i]

    def _select_second_index_svr(self, i, m, X, y, alpha, alpha_star, b, epsilon):
        max_step = 0
        selected_j = -1
        E_i = self._compute_error_svr(i, X, y, alpha, alpha_star, b, epsilon)
        
        for j in range(m):
            if j == i:
                continue
            
            E_j = self._compute_error_svr(j, X, y, alpha, alpha_star, b, epsilon)
            step = abs(E_i - E_j)
            
            if step > max_step:
                max_step = step
                selected_j = j
        
        return selected_j if selected_j != -1 else (i + 1) % m

    def fit(self, X, y):
        m, n = X.shape
        alpha = np.zeros(m)
        alpha_star = np.zeros(m)
        b = 0
        passes = 0
        
        while passes < self.max_passes:
            num_changed_alphas = 0
            
            for i in range(m):
                E_i = self._compute_error_svr(i, X, y, alpha, alpha_star, b, self.epsilon)
                if (abs(E_i) > self.epsilon + self.tol and alpha[i] < self.C) or (abs(E_i) < self.epsilon - self.tol and alpha_star[i] < self.C):
                    j = self._select_second_index_svr(i, m, X, y, alpha, alpha_star, b, self.epsilon)
                    E_j = self._compute_error_svr(j, X, y, alpha, alpha_star, b, self.epsilon)
                    
                    alpha_i_old, alpha_star_i_old = alpha[i], alpha_star[i]
                    alpha_j_old, alpha_star_j_old = alpha[j], alpha_star[j]
                    
                    if abs(y[i] - y[j]) > self.epsilon:
                        L = max(0, alpha[j] - alpha_star[i])
                        H = min(self.C, self.C + alpha[j] - alpha_star[i])
                    else:
                        L = max(0, alpha_star[i] + alpha[j] - self.C)
                        H = min(self.C, alpha_star[i] + alpha[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * X[i, :] @ X[j, :].T - X[i, :] @ X[i, :].T - X[j, :] @ X[j, :].T
                    if eta >= 0:
                        continue
                    
                    alpha[j] -= (E_i - E_j) / eta
                    alpha[j] = np.clip(alpha[j], L, H)
                    
                    if abs(alpha[j] - alpha_j_old) < self.tol:
                        continue
                    
                    alpha_star[i] += (alpha_star_i_old - alpha_star[j])
                    
                    b1 = b - E_i - alpha[i] * (alpha[i] - alpha_i_old) * X[i, :] @ X[i, :].T - alpha[j] * (alpha[j] - alpha_j_old) * X[i, :] @ X[j, :].T
                    b2 = b - E_j - alpha_star[i] * (alpha_star[i] - alpha_star_i_old) * X[j, :] @ X[j, :].T - alpha_star[j] * (alpha_star[j] - alpha_star_j_old) * X[j, :] @ X[j, :].T
                    
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha_star[i] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        w = np.sum((alpha - alpha_star)[:, np.newaxis] * X, axis=0)

        self.w = w
        self.b = b




    def predict(self, X):
        return X @ self.w + self.b
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)






