from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from hw import Jamshidian
from hw import Henrard
from numpy import exp
from hw import hw_helper

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
    
    
    def generate_paths(self, until, timestep, spotstep):
        dt = until / timestep
        
        noise = np.random.normal(0, 1., (timestep, spotstep))
        diffusion = np.zeros([timestep+1, spotstep]) + self.initial_value
        
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
    
    
    def generate_paths(self, until, timestep, spotstep):
        
        time_ax = np.linspace(0, until, timestep+1) # +1 to have pretty sampling
        dt = until/timestep
        
        noise = np.random.normal(0, 1., (timestep, spotstep))
        
        euler_diffusion = np.zeros([timestep+1, spotstep]) + self.initial_value
        euler_diffusion_no_noise = np.zeros([timestep+1, 1]) + self.initial_value
        
        exact_diffusion_no_noise = np.zeros([timestep+1, 1]) + self.initial_value
        numerical_drift_diffusion = np.zeros([timestep+1, 1]) + self.initial_value
        exact_diffusion = np.zeros([timestep+1, spotstep]) + self.initial_value

        # EULER
        for (t_idx, t), rnd in zip(enumerate(time_ax), noise):
                euler_diffusion[t_idx+1] = (euler_diffusion[t_idx]
                                      - self.isT * Jamshidian._B(t, until, self.mr) * self.sigma(t)**2 * dt
                                      - self.mr * euler_diffusion[t_idx] * dt 
                                      + self.sigma(t) * np.sqrt(dt) * rnd)
                
                
        for t_idx, t in enumerate(time_ax[:-1]):
            euler_diffusion_no_noise[t_idx+1] = (euler_diffusion_no_noise[t_idx]
                                      - self.isT * Jamshidian._B(t, until, self.mr) * self.sigma(t)**2 * dt
                                      - self.mr * euler_diffusion_no_noise[t_idx] * dt )
        
        # EXACT SOLUTION
        for t_idx, t in enumerate(time_ax[:-1]):
                exact_diffusion_no_noise[t_idx+1] = (exact_diffusion_no_noise[t_idx] * exp( - self.mr * dt) 
                                            - self.isT * self.drift_T(s=t, t=t+dt, U=until)
                )
                
        for t_idx, t in enumerate(time_ax[:-1]):
                numerical_drift_diffusion[t_idx+1] = (numerical_drift_diffusion[t_idx]
                                            - self.isT * self.drift_T(s=t, t=t+dt, U=until)
                )
        
        for (t_idx, t), rnd in zip(enumerate(time_ax), noise):
                exact_diffusion[t_idx+1] = (exact_diffusion[t_idx] * exp( - self.mr * dt) 
                                            - self.isT * self.drift_T(s=t, t=t+dt, U=until)
                                            + self.sigma(t) * exp(-self.mr * dt) * np.sqrt(dt) * rnd
                )
                
                
        euler_df = pd.DataFrame(euler_diffusion, index = np.linspace(0, until, timestep + 1 ) )
        euler_numerical_drift_df = pd.DataFrame(euler_diffusion_no_noise, index = np.linspace(0, until, timestep + 1 ) )
        
        exact_drift_df = pd.DataFrame(exact_diffusion_no_noise, index = np.linspace(0, until, timestep + 1 ) )
        numerical_drift_df = pd.DataFrame(numerical_drift_diffusion, index = np.linspace(0, until, timestep + 1 ) )
        exact_df = pd.DataFrame(exact_diffusion, index = np.linspace(0, until, timestep + 1 ) )
        
        return euler_df, euler_numerical_drift_df, exact_drift_df, numerical_drift_df, exact_df
    

    def fwd_bond(self, t, T, dsc_curve, x):
        A = Jamshidian._A(t, T, self.mr, self.sigma, dsc_curve)
        B = Jamshidian._B(t, T, self.mr)
        return A * exp(-B * (x + self.isT * self.drift_T(t, T)))
    
    
        




    
    
if __name__ == '__main__':
    print('df')