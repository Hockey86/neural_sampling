"""
Linear regression using neural sampling.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y, check_random_state
#import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('ticks')
from neural_sampler import *


class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, sampler, sigma2_y=1., sigma2_W=1., fit_intercept=True, tau=10.,
                n_draw=1000, n_iter=100, n_chain=1, keep_sampling=0.5,
                random_state=None):
        self.sampler = sampler
        self.sigma2_y = sigma2_y
        self.sigma2_W = sigma2_W
        self.fit_intercept = fit_intercept
        self.tau = tau
        self.n_draw = n_draw
        self.n_iter = n_iter
        self.n_chain = n_chain   # preserved for future use
        self.keep_sampling = keep_sampling
        self.random_state = random_state
        
    def _upost_params(self, X, y):
        return (np.dot(X.T,X)/self.sigma2_y+np.eye(self.D)/self.sigma2_W, np.dot(X.T,y)/self.sigma2_y)
    
    def _dW_log_upost(self, W, use_params, *params):
        """
        d/dW[log(P(D|W)P(W))] = -AW+B
        = -(sigma^-2_y*X.T*X+sigma^-2_W*I)*W+sigma^-2_y*X.T*y
        """
        if use_params:
            A, B = params
        else:
            X, y = params
            A, B = self._upost_params(X, y)
        return B-np.dot(A,W)
    
    def _log_upost(self, W, X, y):
        Y_XW = y-np.dot(X,W)
        return -np.sum(Y_XW*Y_XW)/self.two_sigma2_y-np.sum(W*W)/self.two_sigma2_W
    
    def fit(self, X, y):
        self.fitted_ = False
        X_, y = check_X_y(X,y)
        self.N = X_.shape[0]
        if self.fit_intercept:
            X_ = np.c_[X_, np.ones(self.N)]
        self.D = X_.shape[1]
        self.two_sigma2_y = 2.*self.sigma2_y
        self.two_sigma2_W = 2.*self.sigma2_W
                
        self.Ws = self.sampler.sample(X_,y)
        self.Ws = self.Ws[-int(round(self.keep_sampling*len(self.Ws))):]
        
        self.fitted_ = True
        return self


if __name__=='__main__':
    import pdb
    
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
    
    hmc_sampler = HMCSampler(self._dW_log_upost, self.D, tau=self.tau,
                    n_draw=self.n_draw, n_iter=self.n_iter, n_chain=self.n_chain,
                    upost_params=self._upost_params, log_upost=self._log_upost,
                    compute_H=True, random_state=self.random_state)
    neural_sampler = NeuralSampler()
    
    lr_hmc = BayesianLinearRegression(hmc_sampler,
                sigma2_y=sigma2_y, sigma2_W=sigma2_W,
                fit_intercept=False, tau=100.,
                n_draw=200, n_iter=100, n_chain=1,
                keep_sampling=1., random_state=random_state)
    
    lr_neural = BayesianLinearRegression(neural_sampler,
                sigma2_y=sigma2_y, sigma2_W=sigma2_W,
                fit_intercept=False, tau=100.,
                n_draw=200, n_iter=100, n_chain=1,
                keep_sampling=1., random_state=random_state)
    
    lr_hmc.fit(X, y)
    
    plt.plot(np.array(lr_hmc.Ws))
    plt.show()

