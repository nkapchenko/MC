from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

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


class HullWhiteProcess(Process):

    def __init__(self, rate_curve, mr, sigma, r0):
            self.rate_curve = rate_curve
            self.mr = mr
            self.sigma = sigma
            self.r0 = r0
    @property
    def initial_value(self):
        return self.r0


class GaussianPathGenerator():
    def __init__(self, process, spotstep, timestep):
        self.process = process
        self.spotstep = spotstep
        self.timestep = timestep
        self.noise = np.random.normal(0, 1., (timestep, spotstep))
        self.diffusion = np.zeros([timestep+1, spotstep]) + process.initial_value

    def generate_paths(self, until):
        if isinstance(self.process, BlackScholesProcess):
            dt = until / self.timestep
            for t, rnd in zip(range(self.timestep), self.noise):
                self.diffusion[t+1] = self.diffusion[t]*(1 + self.process.mu*dt + self.process.vol*np.sqrt(dt)*rnd)
            return pd.DataFrame(self.diffusion, index = np.linspace(0, until, self.timestep + 1) )
        elif isinstance(self.process, HullWhiteProcess):
            dt = until / self.timestep
            for t, rnd in zip(range(self.timestep), self.noise):
                self.diffusion[t+1] = self.diffusion[t] - self.process.mr * self.diffusion[t] * dt + self.process.sigma*np.sqrt(dt)*rnd
            return pd.DataFrame(self.diffusion, index = np.linspace(0, until, self.timestep + 1 ) )
        else:
            print(f'self.process type is {type(self.process)} is not expected')

    
    
if __name__ == '__main__':
    print('df')