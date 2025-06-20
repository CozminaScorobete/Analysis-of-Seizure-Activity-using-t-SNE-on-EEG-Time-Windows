

import numpy as np

class TSNEImplementation:

    """ initiate the hyperparameters for the tSNE function
    n_components: target dimensionality (typically 2 or 3)
    perplexity: controls the balance between local/global aspects
    learning_rate, n_iter, early_exaggeration, momentum: optimization settings
    """
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=12.0, momentum=0.8):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.momentum = momentum

    def ShannonEntropy(self, D, beta=1.0):
        """Calculate the Shannon Entropy with proper handling of edge cases"""
        
        # P_{j|i} = exp(-||x_i - x_j||² / (2 * σ²)) →  beta = 1 / (2 * σ²)
        P = np.exp(-D.copy() * beta)  # ← numerator Eq 1
        
        sumP = np.sum(P)              # ← denominator Eq 1
        
        # Handle edge cases more robustly
        if sumP == 0 or not np.isfinite(sumP):
            H = 0
            P = np.zeros_like(D)      # Return zeros instead of D * 0
        else:
            # Shannon Entropy calculation with safety checks
            if sumP > 1e-15:  # Avoid very small denominators
                # Shannon entropy of P_i
                H = np.log(sumP) + beta * np.sum(D * P) / sumP
                # Convert from exponential to probability distribution
                P = P / sumP
            else:
                H = 0
                P = np.zeros_like(D)
        
        # Ensure H is finite
        if not np.isfinite(H):
            H = 0
            
        return H, P

    def _compute_pairwise_affinities(self, X):
        # Number of data points
        n = X.shape[0]
        # Placeholder for P
        P = np.zeros((n, n))

        # Squared norm 
        sum_X = np.sum(X ** 2, axis=1)
        
        # Dij = ||xi||^2 + ||xj||^2 - 2*xi*xj (fixed typo: was -2xi*2xj)
        # Squared Euclidean distance (fixed typo: was "eckidian")
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
        
        # Ensure distances are non-negative (numerical precision fix)
        D = np.maximum(D, 0)

        # Loop over each data point
        for i in range(n):
            # Get distances to all other points (excluding self)
            Di = D[i, np.concatenate((np.arange(0, i), np.arange(i+1, n)))]
            
            # Start with initial beta
            beta = 1.0
            H, thisP = self.ShannonEntropy(Di, beta)

            # Binary search for beta such that perplexity matches
            betamin = -np.inf
            betamax = np.inf
            # Measure how far current entropy is from the desired perplexity 
            Hdiff = H - np.log2(self.perplexity)
            tries = 0
            
            while np.abs(Hdiff) > 1e-5 and tries < 50:
                # Entropy too large => distribution too flat => increase beta
                if Hdiff > 0:
                    betamin = beta
                    beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
                # Entropy too small => distribution too sharp => decrease beta
                else:
                    betamax = beta
                    beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
                
                # Recalculate the entropy
                H, thisP = self.ShannonEntropy(Di, beta)
                Hdiff = H - np.log2(self.perplexity)
                tries += 1
            
            # Store the tuned probability distribution
            P[i, np.concatenate((np.arange(0, i), np.arange(i+1, n)))] = thisP

        return P

    #from asymetric prob to symetric prob
    def _symmetrize_affinities(self, P):
        #book formula
        P = (P + P.T) / (2.0 * P.shape[0])
        return np.maximum(P, 1e-12)

    def _compute_low_dim_affinities(self, Y):
        sum_Y = np.sum(Y ** 2, axis=1)
        num = -2.0 * np.dot(Y, Y.T)
        #calculate the squared pairwise distance
        num = 1.0 / (1.0 + num + sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :])
        #no self-similarity 0 on the diagonal
        np.fill_diagonal(num, 0.0)
        # from exponential to probability distribution for later use
        Q = num / np.sum(num)
        return np.maximum(Q, 1e-12)


    def _compute_gradient(self, P, Q, Y):
        n = Y.shape[0]
        dY = np.zeros_like(Y)
        PQ = P - Q
        for i in range(n):
            diff = Y[i] - Y
            #Computes all pairwise differences and distances from yi to all other yi
            dist_sq = np.sum(diff ** 2, axis=1)
            mult = (1.0 / (1.0 + dist_sq)) * PQ[i]
            #equation 4 
            dY[i] = 4.0 * np.sum((mult[:, np.newaxis] * diff), axis=0)
        return dY

    def fit_transform(self, X):
        n = X.shape[0]
        P = self._compute_pairwise_affinities(X)
        P = self._symmetrize_affinities(P)

        Y = np.random.normal(0, 1e-4, (n, self.n_components))
        iY = np.zeros_like(Y)

        for iter in range(self.n_iter):
            Q = self._compute_low_dim_affinities(Y)
            #uring the first 250 iterations, inflate P values to exaggerate cluster separation early on described in section 3.4
            P_eff = P * self.early_exaggeration if iter < 250 else P
            #gradient eq 5
            dY = self._compute_gradient(P_eff, Q, Y)
            #momentum and gradient update
            m = 0.5 if iter < 20 else self.momentum
            iY = m * iY - self.learning_rate * dY
            Y = Y + iY
            Y -= np.mean(Y, axis=0)
            #print KL divregrente cost
            if (iter + 1) % 100 == 0:
                C = np.sum(P_eff * np.log(P_eff / Q))
                print(f"Iteration {iter+1}, cost: {C:.4f}")

        return Y