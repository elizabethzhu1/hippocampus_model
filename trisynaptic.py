from models.dg import DG_WilsonCowan
from models.ca3 import CA3_WilsonCowan
from models.ca1 import CA1_WilsonCowan
from helpers import set_dg_parameters, set_ca3_parameters, set_ca1_parameters
import matplotlib.pyplot as plt
import numpy as np
import argparse

"""
Simulate the trisynaptic circuit of the hippocampus. 
EC --> DG
DG + EC (II) --> CA3
CA3 + EC (III) --> CA1
"""

def main(theta_osc):
    # initialize models
    dg_pars = set_dg_parameters()

    # pass in EC input to DG
    dg_pars['ext_E'], dg_pars['ext_I'] = entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=theta_osc)
    dg_pars['is_acetylcholine'] = False
    
    dg = DG_WilsonCowan(**dg_pars)

    # simulate DG
    rE_dg, rI_dg = dg.simulate(rE_init=0.2, rI_init=0.1)
    print("=== DG RATES ===")
    print("EXCITATORY: ", rE_dg)
    print("INHIBITORY: ", rI_dg)
    
    # simulate CA3, passing DG + EC as the external input
    ca3_pars = set_ca3_parameters()
    ca3_pars['ext_E'] = rE_dg + entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=True)[0]
    ca3_pars['ext_I'] = rI_dg + entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=True)[1]
    ca3 = CA3_WilsonCowan(**ca3_pars)
    rE_ca3, rI_ca3, _, _ = ca3.simulate(rE_init=0.2, rI_init=0.1)
    print("=== CA3 RATES ===")
    print("EXCITATORY: ", rE_ca3)
    print("INHIBITORY: ", rI_ca3)

    # simulate CA1, passing CA3 as the external input
    ca1_pars = set_ca1_parameters()
    ca1_pars['ext_E'] = rE_ca3 + entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=True)[0]
    ca1_pars['ext_I'] = rI_ca3 + entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=True)[1]
    ca1 = CA1_WilsonCowan(**ca1_pars)
    rE_ca1, rI_ca1 = ca1.simulate(rE_init=0.2, rI_init=0.1)
    print("=== CA1 RATES ===")
    print("EXCITATORY: ", rE_ca1)
    print("INHIBITORY: ", rI_ca1)

    # plot rate results
    plot_rates(rE_dg, rI_dg, rE_ca3, rI_ca3, rE_ca1, rI_ca1, dg_pars)


def entorhinal_inputs(T, dt, theta_oscillation=False):
    """
    Model external input from Entorhinal Cortext (EC) to DG, CA3, and CA1.
    """
    if theta_oscillation:
        # model EC input as a theta oscillation
        frequency = 5  # HZ
        frequency_ms = frequency / 1000.0  # Convert to cycles per millisecond
        timesteps = int(T // dt)
        ext_E =  0.5 * np.sin(2 * np.pi * frequency_ms * np.arange(0, T, dt)) + 1 + np.random.normal(0, 0.05, timesteps + 1)
        ext_I = 0.5 * np.sin(2 * np.pi * frequency_ms * np.arange(0, T, dt)) + 1 + np.random.normal(0, 0.05, timesteps + 1)
    else:
        timesteps = int(T // dt)
        ext_E = [2] * timesteps + np.random.normal(0, 0.1, timesteps)
        ext_I = [1] * timesteps + np.random.normal(0, 0.1, timesteps)

    return ext_E, ext_I


def plot_rates(rE_dg, rI_dg, rE_ca3, rI_ca3, rE_ca1, rI_ca1, dg_pars):
        # Create time array for plotting
    time_array = np.arange(0, dg_pars['T'], dg_pars['dt'])
    
    # Plot results with proper time axis
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Excitatory rates over time
    plt.subplot(2, 2, 1)
    plt.plot(time_array, rE_dg, label='DG', linewidth=2)
    plt.plot(time_array, rE_ca3, label='CA3', linewidth=2)
    plt.plot(time_array, rE_ca1, label='CA1', linewidth=2)
    plt.ylim(0, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Excitatory Rate')
    plt.title('Excitatory Activity Across Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Theta input to DG
    plt.subplot(2, 2, 2)
    ec_input_e, ec_input_i = entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=True)
    plt.plot(time_array, ec_input_e, label='EC → E', linewidth=2, color='blue')
    plt.plot(time_array, ec_input_i, label='EC → I', linewidth=2, color='red')
    plt.ylim(0, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Amplitude')
    plt.title('Theta Oscillation Input')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: All rates (E and I) for each region
    plt.subplot(2, 2, 3)
    plt.plot(time_array, rE_dg, label='DG E', linewidth=2)
    plt.plot(time_array, rI_dg, label='DG I', linewidth=2, linestyle='--')
    plt.ylim(0, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity Rate')
    plt.title('DG Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: CA3 and CA1 comparison
    plt.subplot(2, 2, 4)
    plt.plot(time_array, rE_ca3, label='CA3 E', linewidth=2)
    plt.plot(time_array, rI_ca3, label='CA3 I', linewidth=2, linestyle='--')
    plt.plot(time_array, rE_ca1, label='CA1 E', linewidth=2)
    plt.plot(time_array, rI_ca1, label='CA1 I', linewidth=2, linestyle='--')
    plt.ylim(0, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity Rate')
    plt.title('CA3 and CA1 Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # plot separtely
    # plt.plot(rE_dg, label='DG')
    # plt.show()
    # plt.plot(rE_ca3, label='CA3')
    # plt.show()
    # plt.plot(rE_ca1, label='CA1')
    # plt.show()
    

if __name__ == "__main__":

    # take arguments
    parser = argparse.ArgumentParser(description='Simulate the trisynaptic circuit of the hippocampus.')
    parser.add_argument('--theta_osc', action='store_true', help='Use theta oscillation input')
    args = parser.parse_args()

    main(args.theta_osc)
