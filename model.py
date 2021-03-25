import numpy as np
from cvxopt import matrix, solvers

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class ClassificationBaseModel: 

    def __init__(self,kernel): 
        """
        Args: 
        - Kernel: python callable fct

        Notes: 
        BaseModel is based on representer theorem. Function fit has to compute the kernel_params array and set is_fitted to True once it is done.
        """
        self.kernel = kernel
        self.is_fitted = False
        self.kernel_params = None
        self.dataset = None
        self.label = None


    def fit(self,X,y): 
        self.dataset = X
        if np.unique(y).sum() == 1:
            self.label = 2*y - 1
        else:
            self.label = y
            
    def predict(self, X): 
        if self.is_fitted: 
            predictions = np.sum(self.comparison_matrix(X, self.dataset)@np.diag(self.kernel_params), axis=1)
            return (np.sign(predictions) + 1)/2
        else: 
            print('Model not fitted')
            
    def score(self, X, y): 
        n_samples = y.shape[0]
        predictions = self.predict(X)
        score = 1 - np.sum(np.abs(predictions-y))/n_samples
        return score, predictions
    
    def comparison_matrix(self,X,Y):
        """
        Args: 
        X,Y,: np.array, first dim number of samples

        Return 
        compmatrix : np.array, comparison matrix, shape : (X.shape[0], Y.shape[0])
        """
        return self.kernel(X, Y)
    
    def gram_matrix(self,X): 
        return self.kernel(X, X)

class KLR(ClassificationBaseModel): 
    def __init__(self,kernel,reg,tol=1e-9):
        super().__init__(kernel)
        self.reg = reg
        self.tol = tol
    
    def fit(self,X,y): 
        super().fit(X,y)
        gram = self.gram_matrix(X)
        self.kernel_params = self.KLR(gram = gram,label = self.label,reg =self.reg,tol = self.tol)
        self.is_fitted = True
        print('Model Fitted')
    
    def predict(self, X): 
        return super().predict(X)
    
    def KLR(self,gram, label, reg, tol):
        n = gram.shape[0]
        assert np.allclose(n, len(label)), 'labels must have the same length as the dimension of the gramian matrix'
        α = np.zeros(n)
        cond = True
        k = 0
        while cond:
            k += 1
            old_α = α.copy()
            m = gram@α
            W = np.diag(np.sqrt(sigmoid(m)*sigmoid(-m)))
            z = m + label/sigmoid(label*m)
            α = W@np.linalg.solve(W@gram@W + n*reg*np.eye(n), W@z)
            cond = (np.linalg.norm(old_α - α) > tol)
        print(k)
        return α

class KSVM(ClassificationBaseModel): 
    def __init__(self,kernel,reg, threshold = None):
        super().__init__(kernel)
        self.reg = reg
        self.threshold = threshold
    
    def fit(self, X, y): 
        super().fit(X,y)
        gram = self.gram_matrix(X)
        self.kernel_params = self.KSVM(gram = gram,label = self.label,reg =self.reg, threshold = self.threshold)
        self.is_fitted = True
        print('Model Fitted')

    def predict(self, X):
        '''
        Arg: 
        X: np.array, first dim number of samples 
        '''
        if self.is_fitted:
            predictions = ((self.label*self.kernel_params[0])@self.comparison_matrix(self.dataset, X) + self.kernel_params[1]).squeeze()
            return (np.sign(predictions) + 1)/2
        else: 
            print('Model not fitted')

    def KSVM(self,gram, label, reg, threshold):
        label = label.squeeze()
        if np.unique(label).sum() == 1:
            label = 2*label - 1
        n = gram.shape[0]
        C = 1/(2*n*reg)
        assert np.allclose(n, len(label)), 'labels must have the same length as the dimension of the gramian matrix'
        labelm = np.diag(label)
        K = matrix(labelm@gram@labelm)
        G = matrix(np.vstack([np.eye(n), -np.eye(n)]))
        A = matrix(label.reshape(1, n))    
        h = matrix(np.vstack([C*np.ones((n, 1)), np.zeros((n, 1))]))
        y = matrix(-np.ones((n, 1)))
        d = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        α = np.array(solvers.qp(K, y, G, h, A, d)['x']).squeeze()
#         α[α < threshold] = 0.
        α[α > C*0.99] = C
        if ((α > C*0.1)*(α < C)).sum() > 0 :
            idx = np.min(np.where(((α > C*0.1)*(α < C)).astype('bool')))
            b = label[idx] - (α*label)@gram[:, idx]
        else:
            b = 0
        return α, b, C
