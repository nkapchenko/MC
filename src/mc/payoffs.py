

"Payoff to apply on mc diffusion"

def call(df_diffusion, K):
    final_state = df_diffusion.iloc[-1]
    
    for final_traj_value in final_state:
        yield (final_traj_value - K) if final_traj_value>K else 0