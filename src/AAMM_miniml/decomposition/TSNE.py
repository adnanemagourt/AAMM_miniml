import numpy as np
from scipy.spatial.distance import pdist, squareform

import sys
sys.path.append('..')
from _BaseClasses import Transformer


class TSNE(Transformer):
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
    
    def _hbeta(self, D, beta):
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P
    
    def _binary_search(self, D, target_perplexity):
        beta = 1.0
        H, P = self._hbeta(D, beta)
        H_diff = H - np.log(target_perplexity)
        
        beta_min = -np.inf
        beta_max = np.inf
        
        iter_count = 0
        while np.abs(H_diff) > 1e-5 and iter_count < 50:
            if H_diff > 0:
                beta_min = beta
                if beta_max == np.inf:
                    beta = beta * 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta = beta / 2.0
                else:
                    beta = (beta + beta_min) / 2.0
            
            H, P = self._hbeta(D, beta)
            H_diff = H - np.log(target_perplexity)
            iter_count += 1
        
        return P
    
    def _compute_joint_probabilities(self, X):
        n = X.shape[0]
        D = squareform(pdist(X, "sqeuclidean"))
        P = np.zeros((n, n))
        target_perplexity = self.perplexity
        
        for i in range(n):
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = self._binary_search(D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))], target_perplexity)
        
        P = (P + P.T) / (2.0 * n)
        return P
    
    def fit_transform(self, X):
        n = X.shape[0]
        momentum = 0.5
        final_momentum = 0.8
        eta = self.learning_rate
        
        P = self._compute_joint_probabilities(X) * 4.0
        P = np.maximum(P, 1e-12)
        
        Y = np.random.randn(n, self.n_components)
        dY = np.zeros((n, self.n_components))
        iY = np.zeros((n, self.n_components))
        
        for iter in range(self.n_iter):
            sum_Y = np.sum(np.square(Y), axis=1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            np.fill_diagonal(num, 0.)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)
            
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.n_components, 1)).T * (Y[i, :] - Y), 0)
            
            if iter < 20:
                momentum = momentum
            else:
                momentum = final_momentum
            
            iY = momentum * iY - eta * dY
            Y = Y + iY
            Y = Y - np.mean(Y, axis=0)
            
            if (iter + 1) % 100 == 0:
                C = np.sum(P * np.log(P / Q))
                print(f"Iteration {iter+1}: error is {C}")
            
            if iter == 100:
                P = P / 4.
        
        return Y

# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Sample DataFrame
    data = {
        'Feature1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
        'Feature2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
    }
    df = pd.DataFrame(data)

    # Create t-SNE instance
    tsne = TSNE(n_components=1, perplexity=10.0, n_iter=300, learning_rate=0.1)

    # Transform the DataFrame
    df_reduced_tsne = tsne.fit_transform(df.values)

    print("t-SNE Reduced DataFrame:\n", df_reduced_tsne)