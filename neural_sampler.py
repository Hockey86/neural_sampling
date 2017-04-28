"""
Implements general neural sampler,
where the probabilities can be replaced by specific models.
"""
import numpy as np
from sklearn.utils import check_array, check_X_y, check_random_state
#import pymc3 as pm


class NeuralSampler(object):
    """
    NeuralSampler(log_unnorm_posterior, n_draw=1000, n_iter=100, n_chain=1, random_state=None)
    log_unnorm_posterior: a function to compute the log of unnorm posterior: log[P(D|theta)P(theta)]
    n_draw: number of samples drawn
    n_iter: number of leapfrog steps in Hamiltonian dynamics
    n_chain: number of simultaneously sampling chains
    random_state: seed of random number generator, default None.
    """
    def __init__(self, log_unnorm_posterior, n_draw=1000, n_iter=100, n_chain=1, random_state=None):
        self.log_unnorm_posterior = log_unnorm_posterior
        self.n_draw = n_draw
        self.n_iter = n_iter
        self.n_chain = n_chain   # preserved for future use
        self.random_state = random_state

    def sample(self, X, y=None):
        self.random_state_ = check_random_state(self.random_state)
        X,y = check_X_y(X,y)

        self.u = np.zeros()

        for i in range(self.n_draw):
            for j in range(self.n_iter):
                self.v = self.random_state_.normal()
                self.v = self.v - ??


if __name__=='__main__':
    import pdb
