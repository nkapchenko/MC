import numpy as np
import pandas as pd
from numpy import exp
from hw import hw_helper

"This is optional module which demonstrates some features"

def mc_curiosity(hw_process, until, timestep, paths, time_ax , dt, noise):
    """
    return 4 dataframes with the following info:
    euler_df - matrix (dataframe) with monte carlo simulation using eurel discretization of SDE
    euler_drift_df - array (dataframe) with determenistic numeric euler drift
    exact_drift_df - array (dataframe) with determenistic numeric EXACT drift (solution of Orn.-Uhn.)
    drift_T_cumsum_df - cummulative sum of drift_T function (aka M(s,T) : 2nd determenistic term in Orn-Uhn SDE solution under T measure)
    """
    
    mr = hw_process.mr
    sigma = hw_process.sigma
    
    isT = False if hw_process.measure == 0. else True
    
    euler_diffusion = np.zeros([timestep+1, paths]) + hw_process.initial_value
    euler_drift = np.zeros([timestep+1, 1]) + hw_process.initial_value

    exact_drift = np.zeros([timestep+1, 1]) + hw_process.initial_value
    drift_T_cumsum = np.zeros([timestep+1, 1]) + hw_process.initial_value


    # EULER (formula 1.2 wiki)
    for (t_idx, t), rnd in zip(enumerate(time_ax), noise):
            euler_diffusion[t_idx+1] = (euler_diffusion[t_idx]
                                  - isT * hw_helper._B(t, until, mr) * sigma(t)**2 * dt
                                  - mr * euler_diffusion[t_idx] * dt 
                                  + sigma(t) * np.sqrt(dt) * rnd)

    # EULER NUMERIC DRIFT (no noise)       
    for t_idx, t in enumerate(time_ax[:-1]):
        euler_drift[t_idx+1] = (euler_drift[t_idx]
                                  - isT * hw_helper._B(t, until, mr) * sigma(t)**2 * dt
                                  - mr * euler_drift[t_idx] * dt )

    # EXACT SOLUTION NUMERIC DRIFT (no noise)
    for t_idx, t in enumerate(time_ax[:-1]):
            exact_drift[t_idx+1] = (hw_process.expectation(s=t, x0=exact_drift[t_idx], dt=dt))


    # CUMSUM OF T-measure drift term (real curiosity)       
    for t_idx, t in enumerate(time_ax[:-1]):
            drift_T_cumsum[t_idx+1] = (drift_T_cumsum[t_idx] - isT * hw_helper.get_drift_T(s=t, t=t+dt, U=until, a=mr, sigma=sigma))

    euler_df, euler_drift_df, exact_drift_df, drift_T_cumsum_df =( 
    [pd.DataFrame(x, index = np.linspace(0, until, timestep + 1 ) ) for x in [euler_diffusion, euler_drift, exact_drift, drift_T_cumsum] ])
    
    return  euler_df, euler_drift_df, exact_drift_df, drift_T_cumsum_df