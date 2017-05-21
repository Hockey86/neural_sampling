"""
Linear regression using neural sampling.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y, check_random_state
#import pymc3 as pm
from neural_sampler import *


class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, sigma2_y=1., sigma2_W=1., fit_intercept=True,
                sampler=None, keep_sampling=0.5, random_state=None):
        self.sigma2_y = sigma2_y
        self.sigma2_W = sigma2_W
        self.fit_intercept = fit_intercept
        self.sampler = sampler
        self.keep_sampling = keep_sampling
        self.random_state = random_state
        
    def _post_params(self, X, y):
        return (np.dot(X.T,X)/self.sigma2_y+np.eye(self.D)/self.sigma2_W, np.dot(X.T,y)/self.sigma2_y)
    
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
        return B-np.dot(A,W)
    
    def _log_post(self, W, X, y):
        """
        log(P(D|W)P(W))] = 
        """
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
        
        if self.sampler is None:
            # we know the posterior
            self.W_mean = None###
            self.W_std = None
        else:
            # use sampler to sample the posterior
            self.sampler.len_theta = self.D
            self.sampler.dtheta_log_post = self._dW_log_post
            self.sampler.post_params = self._post_params
            self.sampler.log_post = self._log_post

            self.W_samples = self.sampler.sample(X_,y)
            self.Ws = self.W_samples[-int(round(self.keep_sampling*len(self.W_samples))):]
        
        self.fitted_ = True
        return self


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
    
    hmc_sampler = HMCSampler(iter_rate=0.01, n_draw=200, n_iter=100,
                     compute_H=True, n_chain=1,
                    random_state=random_state)
    #neural_sampler = NeuralSampler()
    
    lr_hmc = BayesianLinearRegression(sigma2_y=sigma2_y, sigma2_W=sigma2_W,
                sampler=hmc_sampler, keep_sampling=0.5,
                fit_intercept=False, random_state=random_state+1)    
    #lr_neural = BayesianLinearRegression(sigma2_y=sigma2_y, sigma2_W=sigma2_W,
    #            sampler=neural_sampler, keep_sampling=0.5,
    #            fit_intercept=False, random_state=random_state+1)   
    
    lr_hmc.fit(X, y)
    #lr_neural.fit(X, y)
    
    plt.figure()
    plt.plot(np.array(lr_hmc.Ws))
    #plt.figure()
    #plt.plot(np.array(lr_neural.Ws))
    
    sb.despine()
    plt.tight_layout()
    plt.show()

