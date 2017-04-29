"""
Implements general neural sampler,
where the probabilities can be replaced by specific models.
"""
import numpy as np
from sklearn.utils import check_array, check_X_y, check_random_state
#import pymc3 as pm


class NeuralSampler(object):
    """
    NeuralSampler(log_upost, n_draw=1000, n_iter=100, n_chain=1, random_state=None)
    dtheta_log_upost: a function to compute the log of (unnormalized) posterior: log[P(D|theta)P(theta)]
    n_draw: number of samples drawn
    n_iter: number of leapfrog steps in Hamiltonian dynamics
    n_chain: number of simultaneously sampling chains
    random_state: seed of random number generator, default None.
    """
    def __init__(self, dtheta_log_upost, len_theta, tau=10., n_draw=1000,
            n_iter=100, n_chain=1, upost_params=None, log_upost=None,
            compute_H=False, random_state=None):
        self.dtheta_log_upost = dtheta_log_upost
        self.len_theta = len_theta
        self.tau = tau
        self.n_draw = n_draw
        self.n_iter = n_iter
        self.n_chain = n_chain   # preserved for future use
        self.upost_params = upost_params
        self.log_upost = log_upost
        self.compute_H = compute_H
        self.random_state = random_state
        self.Ms = None   # preserved for future use

    def sample(self, X, y=None):
        self.random_state_ = check_random_state(self.random_state)
        #X,y = check_X_y(X,y)
        self.tauinv = 1./self.tau
        self.half_tauinv = 0.5*self.tauinv
        use_params = self.upost_params is not None
        
        if use_params:
            params = self.upost_params(X,y)
        else:
            params = (X,y)

        theta = np.zeros(self.len_theta)
        self.thetas = []
        if self.compute_H:
            self.Hs = []

        for i in range(self.n_draw):
            current_theta = theta
            current_v = self.random_state_.randn(self.len_theta)
            
            v = current_v + self.half_tauinv*self.dtheta_log_upost(theta, use_params, *params)
            for j in range(self.n_iter):
                theta = theta + self.tauinv*v
                if j!=self.n_iter-1:
                    v = v + self.tauinv*self.dtheta_log_upost(theta, use_params, *params)            
            v = v + self.half_tauinv*self.dtheta_log_upost(theta, use_params, *params)
            #v = -v
            
            current_H = -self.log_upost(current_theta, X, y)+np.sum(current_v*current_v)/2.
            proposed_H = -self.log_upost(theta, X, y)+np.sum(v*v)/2.
            if np.log(self.random_state_.rand())<current_H-proposed_H:
                self.thetas.append(theta)  # accept
            else:
                self.thetas.append(current_theta)  # reject
            if self.compute_H:
                self.Hs.append(-self.log_upost(theta, X, y)+np.sum(v*v)/2.)
            
            #v = current_v
            #if self.compute_H:
            #    self.Hs = []
            #for j in range(self.n_iter):
            #    v = v + self.half_tauinv*self.dtheta_log_upost(theta, use_params, *params)
            #    theta = theta + self.tauinv*v
            #    v = v + self.half_tauinv*self.dtheta_log_upost(theta, use_params, *params)
            #    if self.compute_H:
            #        self.Hs.append(-self.log_upost(theta, X, y)+np.sum(v*v)/2.)
            
        
        return self.thetas

