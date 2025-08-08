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

def main(args):
    # initialize models
    dg_pars = set_dg_parameters()

    # pass in EC input to DG
    dg_pars['ext_E'], dg_pars['ext_I'] = entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=args.theta_osc)
    dg_pars['is_acetylcholine'] = args.ach_dg
    
    dg = DG_WilsonCowan(**dg_pars)

    # simulate DG
    rE_dg, rI_dg = dg.simulate(rE_init=0.2, rI_init=0.1)
    print("=== DG RATES ===")
    print("EXCITATORY: ", rE_dg)
    print("INHIBITORY: ", rI_dg)
    
    # simulate CA3, passing DG + EC as the external input
    ca3_pars = set_ca3_parameters()
    ca3_wDG_E = 0.5
    ca3_wDG_I = 0.5
    ca3_wEC_E = 0.5
    ca3_wEC_I = 0.5

    ca3_pars['ext_E'] = ca3_wDG_E * rE_dg + ca3_wEC_E * entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=args.theta_osc)[0]
    ca3_pars['ext_I'] = ca3_wDG_I * rI_dg + ca3_wEC_I * entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=args.theta_osc)[1]
    ca3_pars['is_acetylcholine'] = args.ach_dg
    
    ca3 = CA3_WilsonCowan(**ca3_pars)
    rE_ca3, rI_ca3 = ca3.simulate(rE_init=0.2, rI_init=0.1)
    
    print("=== CA3 RATES ===")
    print("EXCITATORY: ", rE_ca3)
    print("INHIBITORY: ", rI_ca3)

    # simulate CA1, passing CA3 + EC as the external input
    ca1_pars = set_ca1_parameters()
    ca1_wDG_E = 0.5
    ca1_wDG_I = 0.5
    ca1_wEC_E = 0.5
    ca1_wEC_I = 0.5

    ca1_pars['ext_E'] = ca1_wDG_E * rE_ca3 + ca1_wEC_E * entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=args.theta_osc)[0]
    ca1_pars['ext_I'] = ca1_wDG_I * rI_ca3 + ca1_wEC_I * entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=args.theta_osc)[1]
    ca1_pars['is_acetylcholine'] = args.ach_dg
    
    ca1 = CA1_WilsonCowan(**ca1_pars)
    rE_ca1, rI_ca1 = ca1.simulate(rE_init=0.2, rI_init=0.1)
   
    print("=== CA1 RATES ===")
    print("EXCITATORY: ", rE_ca1)
    print("INHIBITORY: ", rI_ca1)

    # plot activity with ACh
    if args.ach_dg:
        fig = dg.plot_activity_with_ach(rE_dg, rI_dg, title="DG Activity with ACh Modulation")
        fig.show()
        fig = ca3.plot_activity_with_ach(rE_ca3, rI_ca3, title="CA3 Activity with ACh Modulation")
        fig.show()
        fig = ca1.plot_activity_with_ach(rE_ca1, rI_ca1, title="CA1 Activity with ACh Modulation")
        fig.show()

    # plot rate results
    if args.ach_dg:
        print("ACH")
        ach_trace = []
        for t in range(len(rE_dg)):
            ach_trace.append(dg.ACh_func(t))
    else:
        # no ACh added
        ach_trace = [0] * len(rE_dg)

    plot_rates(rE_dg, rI_dg, rE_ca3, rI_ca3, rE_ca1, rI_ca1, dg_pars, args.theta_osc, ach_trace)


def entorhinal_inputs(T, dt, theta_oscillation=False):
    """
    Model external input from Entorhinal Cortext (EC) to DG, CA3, and CA1.
    """
    timesteps = np.arange(0, T, dt)
    
    if theta_oscillation:
        # model EC input as a theta oscillation
        frequency = 5  # HZ
        frequency_ms = frequency / 1000.0  # Convert to cycles per millisecond
        ext_E = 0.5 * np.sin(2 * np.pi * frequency_ms * timesteps) + 1 + np.random.normal(0, 0.05, len(timesteps))
        ext_I = 0.5 * np.sin(2 * np.pi * frequency_ms * timesteps) + 1 + np.random.normal(0, 0.05, len(timesteps))
    else:
        ext_E = np.ones(len(timesteps)) + np.random.normal(0, 0.1, len(timesteps))
        ext_I = np.ones(len(timesteps)) + np.random.normal(0, 0.1, len(timesteps))

    return ext_E, ext_I


def plot_rates(rE_dg, rI_dg, rE_ca3, rI_ca3, rE_ca1, rI_ca1, dg_pars, theta_osc, ach_trace):
    # Create time array for plotting
    time_array = np.arange(0, dg_pars['T'], dg_pars['dt'])
    
    # Plot results with proper time axis
    if ach_trace is not None:
        plt.figure(figsize=(10, 10)) # Make figure taller to accommodate ACh plot
        n_plots = 5
    else:
        plt.figure(figsize=(10, 8))
        n_plots = 4
    
    # Plot 1: Excitatory rates over time
    plt.subplot(3, 2, 1)
    plt.plot(time_array, rE_dg, label='DG', linewidth=2)
    plt.plot(time_array, rE_ca3, label='CA3', linewidth=2)
    plt.plot(time_array, rE_ca1, label='CA1', linewidth=2)
    plt.ylim(0, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Excitatory Rate')
    plt.title('Excitatory Activity Across Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    # Plot 2: Theta input to DG
    plt.subplot(3, 2, 2)
    ec_input_e, ec_input_i = entorhinal_inputs(dg_pars['T'], dg_pars['dt'], theta_oscillation=theta_osc)
    plt.plot(time_array, ec_input_e, label='EC → E', linewidth=2, color='blue')
    plt.plot(time_array, ec_input_i, label='EC → I', linewidth=2, color='red')
    plt.ylim(0, 3)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Amplitude')
    plt.title('Theta Oscillation Input')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    # Plot 3: All rates (E and I) for each region
    plt.subplot(3, 2, 3)
    plt.plot(time_array, rE_dg, label='DG E', linewidth=2)
    plt.plot(time_array, rI_dg, label='DG I', linewidth=2, linestyle='--')
    plt.ylim(0, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity Rate')
    plt.title('DG Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    # Plot 4: CA3 and CA1 comparison
    plt.subplot(3, 2, 4)
    plt.plot(time_array, rE_ca3, label='CA3 E', linewidth=2, color='green')
    plt.plot(time_array, rI_ca3, label='CA3 I', linewidth=2, linestyle='--', color='red')
    plt.plot(time_array, rE_ca1, label='CA1 E', linewidth=2, color='blue')
    plt.plot(time_array, rI_ca1, label='CA1 I', linewidth=2, linestyle='--', color='pink')
    plt.ylim(0, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity Rate')
    plt.title('CA3 and CA1 Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    # Plot 5: Acetylcholine in DG (if added)
    if ach_trace is not None:
        plt.subplot(3, 2, 5)  # Changed from 2,2,5 to 3,2,5
        plt.plot(time_array, ach_trace, label='DG ACh', linewidth=2, color='purple')
        plt.ylim(0, 1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Acetylcholine Level')
        plt.title('Acetylcholine in DG')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=1.0, w_pad=1.0)
    
    plt.show()

if __name__ == "__main__":

    # take arguments
    parser = argparse.ArgumentParser(description='Simulate the trisynaptic circuit of the hippocampus.')
    parser.add_argument('--theta_osc', action='store_true', help='Use theta oscillation input')
    parser.add_argument('--ach_dg', action='store_true', help='Add acetylcholine to DG')
    args = parser.parse_args()

    main(args)
