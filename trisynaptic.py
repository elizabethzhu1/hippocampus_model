from models.dg import DG_WilsonCowan
from models.ca3 import CA3_WilsonCowan
from models.ca1 import CA1_WilsonCowan
from helpers import set_dg_parameters, set_ca3_parameters, set_ca1_parameters
import matplotlib.pyplot as plt
import numpy as np

"""
Simulate the trisynaptic circuit of the hippocampus. 
EC --> DG
DG + EC (II) --> CA3
CA3 + EC (III) --> CA1
"""

def entorhinal_inputs(T, dt, theta_oscillation=False):
    """
    Model external input from Entorhinal Cortext (EC) to DG, CA3, and CA1.
    """
    if theta_oscillation:
        # model EC input as a theta oscillation
        frequency = 10  # HZ
        timesteps = int(T // dt)
        ext_E = 3 * np.sin(2 * np.pi * frequency * np.arange(0, T, dt)) + 1
        ext_I = 3 * np.sin(2 * np.pi * frequency * np.arange(0, T, dt)) + 1
    else:
        timesteps = int(T // dt)
        ext_E = [2] * timesteps + np.random.normal(0, 0.1, timesteps)
        ext_I = [1] * timesteps + np.random.normal(0, 0.1, timesteps)

    return ext_E, ext_I


def main():
    
    # initialize models
    dg_pars = set_dg_parameters()

    # pass in EC input to DG
    dg_pars['ext_E'], dg_pars['ext_I'] = entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=True)
    dg_pars['is_acetylcholine'] = False
    
    dg = DG_WilsonCowan(**dg_pars)

    # simulate DG
    rE_dg, rI_dg = dg.simulate(rE_init=0.2, rI_init=0.1)
    print("=== DG RATES ===")
    print("EXCITATORY: ", rE_dg)
    print("INHIBITORY: ", rI_dg)
    
    # simulate CA3, passing DG as the external input
    ca3_pars = set_ca3_parameters()
    ca3_pars['ext_E'] = rE_dg
    ca3_pars['ext_I'] = rI_dg
    ca3 = CA3_WilsonCowan(**ca3_pars)
    rE_ca3, rI_ca3, _, _ = ca3.simulate(rE_init=0.2, rI_init=0.1)
    print("=== CA3 RATES ===")
    print("EXCITATORY: ", rE_ca3)
    print("INHIBITORY: ", rI_ca3)

    # simulate CA1, passing CA3 as the external input
    ca1_pars = set_ca1_parameters()
    ca1_pars['ext_E'] = rE_ca3
    ca1_pars['ext_I'] = rI_ca3
    ca1 = CA1_WilsonCowan(**ca1_pars)
    rE_ca1, rI_ca1 = ca1.simulate(rE_init=0.2, rI_init=0.1)
    print("=== CA1 RATES ===")
    print("EXCITATORY: ", rE_ca1)
    print("INHIBITORY: ", rI_ca1)

    # plot results
    plt.plot(rE_dg, label='DG')
    plt.plot(rE_ca3, label='CA3')
    plt.plot(rE_ca1, label='CA1')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()


"""
Scratchpad

- add acetylcholine to circuit (can release in all regions or individual regions)

"""
