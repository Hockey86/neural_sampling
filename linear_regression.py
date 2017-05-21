"""
Linear regression using neural sampling.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y, check_random_state
#import pymc3 as pm
#from neural_sampler import *


class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, sigma2_y=1., sigma2_W=1.,
            fit_intercept=True, random_state=None):
        self.sigma2_y = sigma2_y
        self.sigma2_W = sigma2_W
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        
    def _post_params(self, X, y):
        return (-np.dot(X.T,X)/self.sigma2_y-np.eye(self.D)/self.sigma2_W, np.dot(X.T,y)/self.sigma2_y)
    
    def fit(self, X, y):
        self.fitted_ = False
                    
        X_, y = check_X_y(X,y)
        self.N = X_.shape[0]
        if self.fit_intercept:
            X_ = np.c_[X_, np.ones(self.N)]
        self.D = X_.shape[1]
        self.two_sigma2_y = 2.*self.sigma2_y
        self.two_sigma2_W = 2.*self.sigma2_W
        
        # we know the posterior
        self.W_mean = None###
        self.W_std = None
        
        self.fitted_ = True
        return self
        
    def predict(self, X):
        pass
        #if hasattr(self, 'W_mean'):
        #else:
        #return yp, yp_std
        
        
class BayesianLinearRegressionHMC(BayesianLinearRegression):
    """
    BayesianLinearRegressionHMC(sigma2_y=1., sigma2_W=1., fit_intercept=True,
                iter_rate=0.01, n_draw=1000, n_iter=100, n_chain=1,
                compute_H=False, keep_sampling=0.5, random_state=None)
    n_draw: number of samples drawn
    n_iter: number of leapfrog steps in Hamiltonian dynamics
    n_chain: number of simultaneously sampling chains
    random_state: seed of random number generator, default None.
    """
    def __init__(self, sigma2_y=1., sigma2_W=1., fit_intercept=True,
                iter_rate=0.01, n_draw=1000, n_iter=100, n_chain=1,
                compute_H=False, keep_sampling=0.5, random_state=None):
        super(BayesianLinearRegressionHMC, self).__init__(sigma2_y=sigma2_y,
                sigma2_W=sigma2_W, fit_intercept=fit_intercept,
                random_state=random_state)
        self.iter_rate = iter_rate
        self.n_draw = n_draw
        self.n_iter = n_iter
        self.n_chain = n_chain   # preserved for future use
        self.compute_H = compute_H
        self.keep_sampling = keep_sampling
        self.Ms = None   # preserved for future use
    
    def _dW_log_post(self, W, use_params, *params):
        """
        d/dW[log(P(D|W)P(W))] = AW+B
        = -(sigma^-2_y*X.T*X+sigma^-2_W*I)*W+sigma^-2_y*X.T*y
        """
        if use_params:
            A, B = params
        else:
            X, y = params
            A, B = self._post_params(X, y)
        return np.dot(A,W)+B
    
    def _log_post(self, W, X, y):
        """
        log(P(D|W)P(W))] = 
        """
        Y_XW = y-np.dot(X,W)
        return -np.sum(Y_XW*Y_XW)/self.two_sigma2_y-np.sum(W*W)/self.two_sigma2_W
        
    def _sample(self, X, y=None):
        use_params = self._post_params is not None        
        if use_params:
            params = self._post_params(X,y)
        else:
            params = (X,y)

        W = np.zeros(self.D)
        self.W_samples = []
        if self.compute_H:
            self.Hs = []

        for i in range(self.n_draw):
            current_W = W
            current_v = self.random_state_.randn(self.D)
            
            v = current_v + self.half_iter_rate*self._dW_log_post(W, use_params, *params)
            for j in range(self.n_iter):
                W = W + self.iter_rate*v
                if j!=self.n_iter-1:
                    v = v + self.iter_rate*self._dW_log_post(W, use_params, *params)            
            v = v + self.half_iter_rate*self._dW_log_post(W, use_params, *params)
            #v = -v
            
            current_H = -self._log_post(current_W, X, y)+np.sum(current_v*current_v)/2.
            proposed_H = -self._log_post(W, X, y)+np.sum(v*v)/2.
            if np.log(self.random_state_.rand())<current_H-proposed_H:
                self.W_samples.append(W)  # accept
            else:
                self.W_samples.append(current_W)  # reject
                W = current_W
            if self.compute_H:
                self.Hs.append(-self._log_post(W, X, y)+np.sum(v*v)/2.)
    
    def fit(self, X, y):
        self.fitted_ = False
        self.random_state_ = check_random_state(self.random_state)
                    
        X_, y = check_X_y(X,y)
        self.N = X_.shape[0]
        if self.fit_intercept:
            X_ = np.c_[X_, np.ones(self.N)]
        self.D = X_.shape[1]
        self.two_sigma2_y = 2.*self.sigma2_y
        self.two_sigma2_W = 2.*self.sigma2_W
        self.half_iter_rate = 0.5*self.iter_rate

        self._sample(X_,y)
        self.Ws = self.W_samples[-int(round(self.keep_sampling*len(self.W_samples))):]
        
        self.fitted_ = True
        return self
        
        
class BayesianLinearRegressionNS(BayesianLinearRegressionHMC):
    """
    BayesianLinearRegressionNS(sigma2_y=1., sigma2_W=1., fit_intercept=True,
                tau=10., n_draw=1000, n_iter=100, n_chain=1,
                compute_H=False, keep_sampling=0.5, random_state=None)
    n_draw: number of samples drawn
    n_iter: number of leapfrog steps in Hamiltonian dynamics
    n_chain: number of simultaneously sampling chains
    random_state: seed of random number generator, default None.
    """
    def __init__(self, sigma2_y=1., sigma2_W=1., fit_intercept=True,
                tau=10., pair_n_conn=100, n_draw=1000, n_iter=100, n_chain=1,
                compute_H=False, keep_sampling=0.5, random_state=None):
        super(BayesianLinearRegressionNS, self).__init__(sigma2_y=sigma2_y,
                sigma2_W=sigma2_W, fit_intercept=fit_intercept,
                random_state=random_state)
        self.tau = tau
        self.pair_n_conn = pair_n_conn
        self.n_draw = n_draw
        self.n_iter = n_iter
        self.n_chain = n_chain   # preserved for future use
        self.compute_H = compute_H
        self.keep_sampling = keep_sampling
        self.Ms = None   # preserved for future use
    
    def _dW_log_post(self, W, use_params, *params):
        """
        d/dW[log(P(D|W)P(W))] = -AW+B
        = -(sigma^-2_y*X.T*X+sigma^-2_W*I)*W+sigma^-2_y*X.T*y
        """
        if use_params:
            A, B = params
        else:
            X, y = params
            A, B = self._post_params(X, y)
        
        # spikes: pair_n_conn x D
        spikes_prob = np.tile(np.tanh(np.abs(W)), (self.pair_n_conn,1))
        spikes = (self.random_state_.rand(*spikes_prob.shape)<=spikes_prob).astype(float)
        spikes = spikes*np.sign(W)
                
        #return np.dot(A,W)+B
        #return np.dot(A, spikes.mean(axis=0)) + B
        return np.dot(A, np.arctanh(np.clip(spikes.mean(axis=0),1e-10-1,1-1e-10))) + B
            
    def _log_post(self, W, X, y):
        """
        log(P(D|W)P(W))] = 
        """
        Y_XW = y-np.dot(X,W)
        return -np.sum(Y_XW*Y_XW)/self.two_sigma2_y-np.sum(W*W)/self.two_sigma2_W
        
    def _sample(self, X, y=None):
        use_params = self._post_params is not None        
        if use_params:
            params = self._post_params(X,y)
        else:
            params = (X,y)

        W = np.zeros(self.D)
        self.W_samples = []
        if self.compute_H:
            self.Hs = []

        for i in range(self.n_draw):
            current_W = W
            current_v = self.random_state_.randn(self.D)
            
            v = current_v + self.half_iter_rate*self._dW_log_post(W, use_params, *params)
            for j in range(self.n_iter):
                W = W + self.iter_rate*v
                if j!=self.n_iter-1:
                    v = v + self.iter_rate*self._dW_log_post(W, use_params, *params)            
            v = v + self.half_iter_rate*self._dW_log_post(W, use_params, *params)
            #v = -v
            
            current_H = -self._log_post(current_W, X, y)+np.sum(current_v*current_v)/2.
            proposed_H = -self._log_post(W, X, y)+np.sum(v*v)/2.
            if np.log(self.random_state_.rand())<current_H-proposed_H:
                self.W_samples.append(W)  # accept
            else:
                self.W_samples.append(current_W)  # reject
                W = current_W
            if self.compute_H:
                self.Hs.append(-self._log_post(W, X, y)+np.sum(v*v)/2.)
    
    def fit(self, X, y):
        self.iter_rate = 1./self.tau
        self.half_iter_rate = 0.5*self.iter_rate
        return super(BayesianLinearRegressionNS, self).fit(X,y)


if __name__=='__main__':
    import pdb
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set_style('ticks')
    
    # params
    
    random_state = 2
    np.random.seed(random_state)
    sigma2_y = 0.01
    sigma2_W = 1.
    
    # generate data
    
    N = 100
    D = 3
    W = np.zeros(D)+[2.,1.,3.]
    X = np.random.rand(N,D)
    y = np.dot(X,W)+np.random.randn(N)*np.sqrt(sigma2_y)
    
    #pdb.set_trace()
    lr_neural = BayesianLinearRegressionNS(sigma2_y=sigma2_y, sigma2_W=sigma2_W,
                fit_intercept=False, tau=1000., n_draw=200, n_iter=100,
                n_chain=1, compute_H=True, keep_sampling=0.5,
                random_state=random_state).fit(X, y)
    
    #lr_hmc = BayesianLinearRegressionHMC(sigma2_y=sigma2_y, sigma2_W=sigma2_W,
    #            fit_intercept=False, iter_rate=0.001, n_draw=200, n_iter=100,
    #            n_chain=1, compute_H=True, keep_sampling=0.5,
    #            random_state=random_state).fit(X, y)
    
    plt.figure()
    plt.plot(np.array(lr_neural.Ws))
    sb.despine()
    
    #plt.figure()
    #plt.plot(np.array(lr_neural.Ws))
    #sb.despine()    
    
    plt.figure()
    plt.plot(np.array(lr_neural.Hs))
    sb.despine()
    
    
    plt.tight_layout()
    plt.show()

