# -*- coding: utf-8 -*-

import pytest
from numpy import exp, sqrt, random, mean
from mc import payoffs
from mc import mc
from fox_toolbox.utils import volatility

import numpy.testing as npt

__author__ = "mkapchenko"
__copyright__ = "mkapchenko"
__license__ = "mit"

random.seed(22)

S0 = 100.
K = 100.
T = 10.
r = 0.01
vol = 0.2

tvar = vol*sqrt(T)
F = S0 * exp(r*T)

def test_bs():
    "Check the Black Scholes close formula vs monte carlo diffusion and also with black solution (final distribution)"
    spotstep = 100000
    timestep = 1000
    
    # 1) Black closed formula
    black_formula = volatility.BSPrice(F, K, tvar)
    
    # 2) Black final distribution
    def black_solution(F, K, tvar, x):
        return F * exp(-tvar**2/2 + tvar * x)
    
    final_realisations = [black_solution(F, K, tvar, x) for x in random.normal(0, 1., spotstep)]
    payoff_at_T2 = [(lambda x: x-K if x>K else 0)(x) for x in final_realisations]
    black_distrib = mean(payoff_at_T2)
    
    # 3) Black monte carlo
    black_process = mc.BlackScholesProcess(mu = r, vol = vol, S0 = S0)
    black_diffusion = mc.GaussianPathGenerator(process = black_process, spotstep = spotstep, timestep = timestep)
    
    df_diffusion = black_diffusion.generate_paths(until = T)
    payoff_at_T = list(payoffs.call(df_diffusion, K))
    black_traj = mean(payoff_at_T)
    
    npt.assert_almost_equal(black_formula, black_traj, decimal=1)
    npt.assert_almost_equal(black_formula, black_distrib, decimal=0)

