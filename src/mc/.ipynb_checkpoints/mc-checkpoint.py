from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from numpy import exp
from mc.curiosity import mc_curiosity
from mc.stochasticprocess import StochasticProcess
from collections import namedtuple

from hw import hw_helper



class BlackScholesProcess(StochasticProcess): # should be abc class

    def __init__(self, mu, vol, S0):
        self.mu = mu
        self.vol = vol
        self.S0 = S0

    @property
    def initial_value(self):
        return self.S0
    
    
    def generate_paths(self, until, timestep, paths, curiosity=False):
        dt = until / timestep
        
        noise = np.random.normal(0, 1., (timestep, paths))
        diffusion = np.zeros([timestep+1, paths]) + self.initial_value
        
        for t, rnd in zip(range(timestep), noise):
                diffusion[t+1] = diffusion[t]*(1 + self.mu*dt + self.vol*np.sqrt(dt)*rnd)
                
        return pd.DataFrame(diffusion, index = np.linspace(0, until, timestep + 1 ) )


class HullWhiteProcess(StochasticProcess):

    def __init__(self, model, x0, measure):
        """
        model = hw.HullWhite
        x0 = float or array
        measure - float, 0. for Risk neutral measure, float for T-forward measure
        """
        self.mr = model.mr
        self.sigma = model.sigma
        self.x0 = x0
        self.measure = measure
        
#         self.diffusion = None

            
    @property
    def size(self):
        return 1

                  
    @property
    def initial_value(self):
        return self.x0
    
    @property
    def drift_coef(self, t, x):
        pass
    
    @property
    def diffusion_coef(self, t, x):
        return self.sigma
    
    
    def expectation(self, s, x0, dt):
        """
        compute E( X(s+dt) | Xs = x0)
        s - is a Filtration time Fs (is a lower bound in integral)
        x0 - known state value at time s
        t - is a upper bound in integral (is a time where x(t) is evaluated) 
        """
        
        if self.measure == 0.: # RN
            return  x0 * exp( - self.mr * dt)
        else: # T-forward 
            return x0 * exp( - self.mr * dt) - hw_helper.get_drift_T(s, s+dt, U=self.measure, a=self.mr, sigma=self.sigma)
        
        
    def stdDeviation(self, s, dt):
        return np.sqrt(self.variance(s, dt))
    
    def variance(self, s, dt):
        return hw_helper.get_var_x(T=s+dt, a=self.mr, sigma=self.sigma, s=s)
    
    def covariance(self, s, dt):
        " V(X(s+dt) | Xs = x0) "
        return self.covariance(s, dt)
    
    def evolve(self, s, x0, dt, dw):
        pass
    
    
    def generate_paths(self, until, timestep, spotstep, curiosity = False):
        """
        return dataframe with monte carlo simulation using exact solution for Ornstein Uhlenbeck process (aka Xt).
        
        until - final diffusion time point. If RN=False it defines T-MEASURE maturity (aka U)
        timestep - time discretization number
        spotstep - number of monte carlo trajectories
        
        curiosity - if True returns additional second parameter - tuple with dataframes (for more info check curiosity.mc_curiosity)   
        """
        time_axis = np.linspace(0, until, timestep+1) # +1 to have pretty sampling
        dt = until/timestep
        
        np.random.seed(27)
        noise = np.random.normal(loc=0, scale=1., size=(timestep, spotstep))
        
        exact_diffusion = np.zeros([timestep+1, spotstep]) + self.initial_value
        
        for (t_idx, t), rnd in zip(enumerate(time_axis), noise):
                exact_diffusion[t_idx+1] = ( self.expectation(s=t, x0=exact_diffusion[t_idx], dt=dt) + self.stdDeviation(s=t, dt=dt) * rnd) 
                
        self.diffusion = pd.DataFrame(exact_diffusion, index = time_axis)  
        
        if not curiosity:
            return self.diffusion
        else:       
            return self.diffusion, mc_curiosity(self, until, timestep, spotstep, time_axis , dt, noise)
    
    # TODO: this function should be defined inside HullWhite class
    def fwd_bond(self, t, T, dsc_curve, x): 
        A = hw_helper._A(t, T, self.mr, self.sigma, dsc_curve)
        B = hw_helper._B(t, T, self.mr)
        return A * exp(-B * (x + self.isT * self.drift_T(t, T)))
    
    
        




    
    
if __name__ == '__main__':
    print('df')