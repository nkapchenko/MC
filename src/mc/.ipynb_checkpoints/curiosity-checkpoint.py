import numpy as np
import pandas as pd
from numpy import exp
from hw import hw_helper

def mc_curiosity(hw_process, until, timestep, paths, time_ax , dt, noise):
    """
    return 4 dataframes with the following info:
    euler_df - dataframe with monte carlo simulation using eurel discretization of SDE
    euler_drift_df - 1D dataframe with determenistic numeric euler drift
    exact_drift_df - 1D dataframe with determenistic numeric exact drift from Orn.-Uhn. solution 
    drift_T_cumsum_df - cummulative sum of drift_T function (determenistic term in Orn-Uhn SDE solution under T measure)
    """

    euler_diffusion = np.zeros([timestep+1, paths]) + hw_process.initial_value
    euler_drift = np.zeros([timestep+1, 1]) + hw_process.initial_value

    exact_drift = np.zeros([timestep+1, 1]) + hw_process.initial_value
    drift_T_cumsum = np.zeros([timestep+1, 1]) + hw_process.initial_value


    # EULER
    for (t_idx, t), rnd in zip(enumerate(time_ax), noise):
            euler_diffusion[t_idx+1] = (euler_diffusion[t_idx]
                                  - hw_process.isT * hw_helper._B(t, until, hw_process.mr) * hw_process.sigma(t)**2 * dt
                                  - hw_process.mr * euler_diffusion[t_idx] * dt 
                                  + hw_process.sigma(t) * np.sqrt(dt) * rnd)

    # EULER NUMERIC DRIFT        
    for t_idx, t in enumerate(time_ax[:-1]):
        euler_drift[t_idx+1] = (euler_drift[t_idx]
                                  - hw_process.isT * hw_helper._B(t, until, hw_process.mr) * hw_process.sigma(t)**2 * dt
                                  - hw_process.mr * euler_drift[t_idx] * dt )

    # EXACT SOLUTION NUMERIC DRIFT
    for t_idx, t in enumerate(time_ax[:-1]):
            exact_drift[t_idx+1] = (exact_drift[t_idx] * exp( - hw_process.mr * dt) - hw_process.isT * hw_process.drift_T(s=t, t=t+dt, U=until))

    # CUMSUM OF T-measure drift term       
    for t_idx, t in enumerate(time_ax[:-1]):
            drift_T_cumsum[t_idx+1] = (drift_T_cumsum[t_idx] - hw_process.isT * hw_process.drift_T(s=t, t=t+dt, U=until))

    euler_df, euler_drift_df, exact_drift_df, drift_T_cumsum_df =( 
    [pd.DataFrame(x, index = np.linspace(0, until, timestep + 1 ) ) for x in [euler_diffusion, euler_drift, exact_drift, drift_T_cumsum] ])
    
    return  euler_df, euler_drift_df, exact_drift_df, drift_T_cumsum_df