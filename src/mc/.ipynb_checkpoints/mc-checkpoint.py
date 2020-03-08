from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from numpy import exp
from hw import hw_helper
from mc import curiosity

class Process(metaclass=ABCMeta):
    @abstractmethod
    def initial_value(self):
        pass


class BlackScholesProcess(Process): # should be abc class

    def __init__(self, mu, vol, S0):
        self.mu = mu
        self.vol = vol
        self.S0 = S0

    @property
    def initial_value(self):
        return self.S0
    
    
    def generate_paths(self, until, timestep, paths):
        dt = until / timestep
        
        noise = np.random.normal(0, 1., (timestep, paths))
        diffusion = np.zeros([timestep+1, paths]) + self.initial_value
        
        for t, rnd in zip(range(timestep), noise):
                diffusion[t+1] = diffusion[t]*(1 + self.mu*dt + self.vol*np.sqrt(dt)*rnd)
                
        return pd.DataFrame(diffusion, index = np.linspace(0, until, timestep + 1 ) )


class HullWhiteProcess(Process):
    from hw import Henrard
    from hw import Jamshidian

    def __init__(self, mr, sigma, x0, measure='terminal'):
            self.mr = mr
            self.sigma = sigma
            self.x0 = x0
            self.isT = True if measure =='terminal' else False
            
    @property
    def initial_value(self):
        return self.x0
    
    @property
    def measure(self):
        return 'terminal' if self.isT else 'Risk neutral'
    
    def drift_T(self, s, t, U):
        """s - is a lower bound in integral (is a Filtration time Fs) 
        t - is a upper bound in integral (is a time point where x(t) is evaluated) 
        U - is a measure maturity """
        return hw_helper.get_drift_T(s, t, U, a=self.mr, sigma=self.sigma)
    
    
    def generate_paths(self, until, timestep, paths, curiosity = False):
        """
        return dataframe with monte carlo simulation using exact solution for Ornstein Uhlenbeck process.
        Measure is defined in self.isT
        
        until - final diffusion time point that defines T-MEASURE
        timestep - time discretization number
        paths - number of monte carlo trajectories
        
        curiosity - if True returns additional second parameter - tuple with dataframes (for more info check curiosity.mc_curiosity)   
        """
        
        time_ax = np.linspace(0, until, timestep+1) # +1 to have pretty sampling
        dt = until/timestep
        
        noise = np.random.normal(0, 1., (timestep, paths))
        
        exact_diffusion = np.zeros([timestep+1, paths]) + self.initial_value
        
        for (t_idx, t), rnd in zip(enumerate(time_ax), noise):
                exact_diffusion[t_idx+1] = (exact_diffusion[t_idx] * exp( - self.mr * dt) 
                                            - self.isT * self.drift_T(s=t, t=t+dt, U=until)
                                            + self.sigma(t) * exp(-self.mr * dt) * np.sqrt(dt) * rnd)
                
        exact_df = pd.DataFrame(exact_diffusion, index = np.linspace(0, until, timestep + 1 ) )  
        
        if not curiosity:
            return exact_df
        else:       
            euler_df, euler_drift_df, exact_drift_df, drift_T_cumsum_df = curiosity.mc_curiosity(self, until, timestep, paths, time_ax , dt, noise)
            return exact_df, (euler_df, euler_drift_df, exact_drift_df, drift_T_cumsum_df)
    

    def fwd_bond(self, t, T, dsc_curve, x):
        A = Jamshidian._A(t, T, self.mr, self.sigma, dsc_curve)
        B = Jamshidian._B(t, T, self.mr)
        return A * exp(-B * (x + self.isT * self.drift_T(t, T)))
    
    
        




    
    
if __name__ == '__main__':
    print('df')