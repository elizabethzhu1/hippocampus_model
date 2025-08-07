import numpy as np
import matplotlib.pyplot as plt
from helpers import my_test_plot, plot_nullclines, set_ca3_parameters, get_E_nullcline, get_I_nullcline, plot_bifurcation_diagram_multiple_fps, find_multiple_fixed_points, get_eig_Jacobian, set_ca1_parameters, set_dg_parameters
from models.dg import DG_WilsonCowan
from models.ca3 import CA3_WilsonCowan
from models.ca1 import CA1_WilsonCowan
import argparse


def main(args):

    region = args.region

    print(f"=== {region} Wilson-Cowan Model ===")
    
    if region == 'ca3':
        pars = set_ca3_parameters()
    elif region == 'ca1':
        pars = set_ca1_parameters()
    elif region == 'dg':
        pars = set_dg_parameters()

    # add boolean arguments to pars
    pars['is_adaptation'] = args.is_adaptation
    pars['is_acetylcholine'] = args.is_acetylcholine
    pars['is_theta_modulation'] = args.is_theta_modulation
    pars['is_DG_input'] = args.is_DG_input
    pars['T'] = args.time

    # add a constant noisy external input to pars
    timesteps = int(pars['T'] / pars['dt'])
    
    # add noise to external inputs
    pars['ext_E'] = [args.ext_E] * timesteps + np.random.normal(0, 0.05, timesteps)
    pars['ext_I'] = [args.ext_I] * timesteps + np.random.normal(0, 0.05, timesteps)
    
    if region == 'ca3':
        wc = CA3_WilsonCowan(**pars)
    elif region == 'ca1':
        wc = CA1_WilsonCowan(**pars)
    elif region == 'dg':
        wc = DG_WilsonCowan(**pars)

    # Simulate CA3 model with two initial conditions (one pair is user-specified, the other is 0)
    rE1_ca3, rI1_ca3, _, _ = wc.simulate(rE_init=args.ic_rE, rI_init=args.ic_rI)
    rE2_ca3, rI2_ca3, _, _ = wc.simulate(rE_init=0.0, rI_init=0.0)

    # Create the test plot without showing it
    my_test_plot(pars['range_t'], rE1_ca3, rI1_ca3, rE2_ca3, rI2_ca3, region)

    # Set initial conditions
    rE_init = 0.32
    rI_init = 0.15

    parameter_name = 'wEE'

    # Customize parameter space here
    parameter_space = [8.5, 9, 9.65, 10, 11.4, 12, 12.5, 13, 14, 15, 20]

    all_fixed_points, all_stabilities, parameter_values_list = vary_parameter_and_plot(pars, parameter_name, parameter_space, rE_init, rI_init)

    # plot fixed points in a bifurcation diagram
    if all_fixed_points and all_stabilities and parameter_values_list:
        print(f"\nPlotting bifurcation diagram for {len(parameter_values_list)} parameter values...")
        plot_bifurcation_diagram_multiple_fps(parameter_values_list, all_fixed_points, all_stabilities, parameter_name)

    # plot nullclines for choice of parameters
    Exc_null_rE = np.linspace(-0.01, 0.96, timesteps)
    Inh_null_rI = np.linspace(-0.01, 0.96, timesteps)

    Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)
    Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)
    plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI)
    plt.show()
    

def vary_parameter_and_plot(pars, parameter_name, parameter_values, rE_init, rI_init, model_type='ca3'):
        # Create a figure with subplots in two rows
        n_plots = len(parameter_values)
        n_cols = (n_plots + 1) // 2  # Round up division to get number of columns
        fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 8))
        
        # Flatten axes array for easier indexing
        axes = axes.flatten()

        # First pass to find global y-axis limits
        y_min, y_max = float('inf'), float('-inf')
        for value in parameter_values:
            pars[parameter_name] = value
            wc = CA3_WilsonCowan(**pars)
            rE_std, rI_std, _, _ = wc.simulate(rE_init=rE_init, rI_init=rI_init)
            y_min = min(y_min, min(rE_std), min(rI_std))
            y_max = max(y_max, max(rE_std), max(rI_std))

        # Add some padding to the limits
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        all_fixed_points = []  # List of lists: each inner list contains fixed points for one parameter value
        all_stabilities = []   # List of lists: each inner list contains stabilities for one parameter value
        parameter_values_list = []

        # Plot for each parameter value
        for i, value in enumerate(parameter_values):
            pars[parameter_name] = value
            
            if model_type == 'ca3':
                wc_standard = CA3_WilsonCowan(**pars)
            elif model_type == 'ca1':
                wc_standard = CA1_WilsonCowan(**pars) 
            
            rE_std, rI_std, _, _ = wc_standard.simulate(rE_init=rE_init, rI_init=rI_init)
            
            axes[i].plot(pars['range_t'], rE_std, 'b', label='E population')
            axes[i].plot(pars['range_t'], rI_std, 'r', label='I population')
            axes[i].set_title(f'{parameter_name} = {value}')
            axes[i].set_ylim(y_min, y_max)
            axes[i].legend(loc='best')

            # calculate multiple fixed points
            fps = find_multiple_fixed_points(pars)
            if fps:
                print(f"Found {len(fps)} fixed points for {parameter_name} = {value}")
                all_fixed_points.append(fps)
                all_stabilities.append([])
                parameter_values_list.append(value)
                
                # Check stability of each fixed point
                for fp in fps:
                    eig = get_eig_Jacobian(fp, **pars)
                    is_stable = np.all(eig.real < 0)
                    all_stabilities[-1].append(is_stable)
                    stability_str = "stable" if is_stable else "unstable"
                    print(f"  Fixed point at ({fp[0]:.3f}, {fp[1]:.3f}) is {stability_str}")
            else:
                print(f"No fixed points found for {parameter_name} = {value}")

        # Hide any empty subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.show()

        return all_fixed_points, all_stabilities, parameter_values_list


def vary_multiple_parameters_and_plot(pars, param_dict, rE_init, rI_init, model_type='ca3'):
    """
    Plot dynamics for multiple varying parameters
    
    Args:
        pars: Base parameter dictionary
        param_dict: Dictionary of parameters to vary, e.g.
                    {'wEE': [6.0, 6.1, 6.2], 'wIE': [4.5, 4.6, 4.7]}
    """
    # Get all parameter combinations
    param_names = list(param_dict.keys())
    param_values = list(param_dict.values())
    
    # Create subplot grid
    n_rows = len(param_values[0])  # First parameter determines rows
    n_cols = len(param_values[1])  # Second parameter determines columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # First pass to find global y-axis limits
    y_min, y_max = float('inf'), float('-inf')
    for val1 in param_values[0]:
        for val2 in param_values[1]:
            temp_pars = pars.copy()
            temp_pars[param_names[0]] = val1
            temp_pars[param_names[1]] = val2
            wc = CA3_WilsonCowan(**temp_pars)
            rE, rI = wc.simulate(rE_init=rE_init, rI_init=rI_init)
            y_min = min(y_min, min(rE), min(rI))
            y_max = max(y_max, max(rE), max(rI))

    # Add padding to limits
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # Plot each parameter combination
    for i, val1 in enumerate(param_values[0]):
        for j, val2 in enumerate(param_values[1]):
            temp_pars = pars.copy()
            temp_pars[param_names[0]] = val1
            temp_pars[param_names[1]] = val2
            
            title = f'{param_names[0]}={val1:.2f}\n{param_names[1]}={val2:.2f}'
            
            # Simulate and plot
            if model_type == 'ca3':
                wc = CA3_WilsonCowan(**temp_pars)
            elif model_type == 'ca1':
                wc = CA1_WilsonCowan(**temp_pars) 
        
            rE, rI, _, _ = wc.simulate(rE_init=rE_init, rI_init=rI_init)
            
            axes[i,j].plot(temp_pars['range_t'], rE, 'b', label='E')
            axes[i,j].plot(temp_pars['range_t'], rI, 'r', label='I')
            axes[i,j].set_title(title)
            axes[i,j].set_ylim(y_min, y_max)
            axes[i,j].legend(loc='best')
            
            # Calculate fixed points
            fps = find_multiple_fixed_points(temp_pars)
            if fps:
                print(f"\nFound {len(fps)} fixed points for {title}")
                for fp in fps:
                    eig = get_eig_Jacobian(fp, **temp_pars)
                    is_stable = np.all(eig.real < 0)
                    stability_str = "stable" if is_stable else "unstable"
                    print(f"  Fixed point at ({fp[0]:.3f}, {fp[1]:.3f}) is {stability_str}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Introduce arguments
    parser = argparse.ArgumentParser(description='Simulate a region of the hippocampus with a Wilson-Cowan model')
    parser.add_argument('--region', choices=['ca3', 'ca1', 'dg'], default='ca3', help='Region to simulate')
    
    parser.add_argument('--is_DG_input', action='store_true', help='Include DG input')
    parser.add_argument('--is_acetylcholine', action='store_true', help='Include ACh modulation')
    parser.add_argument('--is_theta_modulation', action='store_true', help='Include theta modulation')
    parser.add_argument('--is_adaptation', action='store_true', help='Include adaptation')
    
    parser.add_argument('--ic_rE', type=float, required=False, default=0.32, help='Specify initial condition for excitatory rate')
    parser.add_argument('--ic_rI', type=float, required=False, default=0.15, help='Specify initial condition for inhibitory rate')

    parser.add_argument('--ext_E', type=float, required=False, default=0.5, help='Specify external input to excitatory population')
    parser.add_argument('--ext_I', type=float, required=False, default=0.5, help='Specify external input to inhibitory population')

    parser.add_argument('--time', type=int, required=False, default=1000, help='Specify number of ms to simulate')

    parser.add_argument('--param', type=str, required=False, default='wEE', choices=['wEE', 'wIE', 'wII', 'wEI', 'a_E', 'theta_E', 'a_I', 'theta_I'], help='Specify parameter to vary')

    args = parser.parse_args()

    main(args)
