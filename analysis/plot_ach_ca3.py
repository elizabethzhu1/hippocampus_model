import numpy as np
import matplotlib.pyplot as plt
from models.ca3 import WilsonCowan
from helpers import set_ca3_parameters, set_parameters

def plot_ca3_with_ach():
    """
    Plot CA3 excitatory-inhibitory activity with ACh modulation
    """
    # Get CA3 parameters
    ca3_pars = set_ca3_parameters()  # customize later
    
    # Create CA3 Wilson-Cowan model
    wc_ca3 = WilsonCowan(**ca3_pars)
    
    # Simulate CA3 model
    rE_ca3, rI_ca3, _, _ = wc_ca3.simulate(rE_init=0.32, rI_init=0.15)
    
    # Plot activity with ACh modulation
    fig = wc_ca3.plot_activity_with_ach(rE_ca3, rI_ca3, "CA3 Activity with ACh Modulation")
    
    # Add some additional analysis
    print("CA3 Model Parameters:")
    print(f"wEE (recurrent excitation): {ca3_pars['wEE']}")
    print(f"wEI (inhibitory to excitatory): {ca3_pars['wEI']}")
    print(f"wIE (excitatory to inhibitory): {ca3_pars['wIE']}")
    print(f"wII (recurrent inhibition): {ca3_pars['wII']}")
    print(f"Initial ACh level: {wc_ca3.ACh_func(0):.3f}")
    print(f"Final ACh level: {wc_ca3.ACh_func(len(ca3_pars['range_t'])-1):.3f}")
    
    plt.show()
    
    return wc_ca3, rE_ca3, rI_ca3

def compare_standard_vs_ca3():
    """
    Compare standard Wilson-Cowan model vs CA3-specific model
    """
    # Get both parameter sets
    std_pars = set_parameters()
    ca3_pars = set_ca3_parameters()
    
    # Create both models
    wc_standard = WilsonCowan(**std_pars)
    wc_ca3 = WilsonCowan(**ca3_pars)
    
    # Simulate both models with same initial conditions
    rE_std, rI_std, _ , _ = wc_standard.simulate(rE_init=0.32, rI_init=0.15)
    rE_ca3, rI_ca3, _ , _ = wc_ca3.simulate(rE_init=0.32, rI_init=0.15)
    
    # Create comparison plot - now 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Standard model plots
    axes[0, 0].plot(std_pars['range_t'], rE_std, 'b', label='Excitatory', linewidth=2)
    axes[0, 0].plot(std_pars['range_t'], rI_std, 'r', label='Inhibitory', linewidth=2)
    axes[0, 0].set_title('Standard Wilson-Cowan Model')
    axes[0, 0].set_ylabel('Firing Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CA3 model plots
    axes[0, 1].plot(ca3_pars['range_t'], rE_ca3, 'b', label='Excitatory', linewidth=2)
    axes[0, 1].plot(ca3_pars['range_t'], rI_ca3, 'r', label='Inhibitory', linewidth=2)
    axes[0, 1].set_title('CA3-Specific Wilson-Cowan Model')
    axes[0, 1].set_ylabel('Firing Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ACh and wEE modulation plot
    ach_values = [wc_ca3.ACh_func(t) for t in range(len(ca3_pars['range_t']))]
    ach_modulation = [wc_ca3.wEE * (1 - ach) for ach in ach_values]
    
    ax_ach = axes[0, 2]
    ax_ach.plot(ca3_pars['range_t'], ach_values, 'g', label='ACh Level', linewidth=2)
    ax_ach_twin = ax_ach.twinx()
    ax_ach_twin.plot(ca3_pars['range_t'], ach_modulation, 'purple', label='Modulated wEE', linewidth=2)
    
    ax_ach.set_xlabel('Time (ms)')
    ax_ach.set_ylabel('ACh Level', color='g')
    ax_ach_twin.set_ylabel('Modulated wEE', color='purple')
    ax_ach.set_title('ACh Modulation and wEE')
    ax_ach.legend(loc='upper right')
    ax_ach_twin.legend(loc='upper left')
    ax_ach.grid(True, alpha=0.3)
    
    # Parameter comparison
    param_names = ['a_E', 'theta_E', 'a_I', 'theta_I', 'wEE', 'wEI', 'wIE', 'wII']
    std_values = [std_pars[name] for name in param_names]
    ca3_values = [ca3_pars[name] for name in param_names]
    
    x = np.arange(len(param_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, std_values, width, label='Standard', alpha=0.8)
    axes[1, 0].bar(x + width/2, ca3_values, width, label='CA3', alpha=0.8)
    axes[1, 0].set_xlabel('Parameters')
    axes[1, 0].set_ylabel('Parameter Values')
    axes[1, 0].set_title('Parameter Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(param_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase space comparison
    axes[1, 1].plot(rE_std, rI_std, 'b', label='Standard', linewidth=2)
    axes[1, 1].plot(rE_ca3, rI_ca3, 'r', label='CA3', linewidth=2)
    axes[1, 1].plot(0.32, 0.15, 'ko', markersize=8, label='Initial Condition')
    axes[1, 1].set_xlabel('Excitatory Rate (rE)')
    axes[1, 1].set_ylabel('Inhibitory Rate (rI)')
    axes[1, 1].set_title('Phase Space Trajectories')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # ACh modulation effect on wEE over time
    time_points = np.arange(len(ca3_pars['range_t']))
    original_wEE = np.full_like(time_points, ca3_pars['wEE'])
    
    axes[1, 2].plot(ca3_pars['range_t'], original_wEE, 'b', label='Original wEE', linewidth=2, linestyle='--')
    axes[1, 2].plot(ca3_pars['range_t'], ach_modulation, 'purple', label='Modulated wEE', linewidth=2)
    axes[1, 2].set_xlabel('Time (ms)')
    axes[1, 2].set_ylabel('wEE Value')
    axes[1, 2].set_title('wEE Modulation by ACh')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)


    
    plt.tight_layout()
    plt.show()
    
    # Print parameter differences
    print("=== Parameter Comparison ===")
    print("Standard vs CA3 Parameters:")
    for name in param_names:
        print(f"{name}: {std_pars[name]:.3f} vs {ca3_pars[name]:.3f}")
    
    # Print ACh modulation info
    print(f"\n=== ACh Modulation Info ===")
    print(f"Original wEE: {ca3_pars['wEE']:.3f}")
    print(f"Initial ACh level: {ach_values[0]:.3f}")
    print(f"Final ACh level: {ach_values[-1]:.3f}")
    print(f"Initial modulated wEE: {ach_modulation[0]:.3f}")
    print(f"Final modulated wEE: {ach_modulation[-1]:.3f}")
    
    return wc_standard, wc_ca3, rE_std, rI_std, rE_ca3, rI_ca3

# def compare_ach_effects():
#     """
#     Compare different ACh modulation effects on CA3 activity
#     """
#     ca3_pars = set_ca3_parameters()
    
#     # Create different ACh scenarios
#     scenarios = [
#         ("High ACh (0.9)", 0.9),
#         ("Medium ACh (0.5)", 0.5), 
#         ("Low ACh (0.1)", 0.1),
#         ("No ACh (0.0)", 0.0)
#     ]
    
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     axes = axes.flatten()
    
#     for i, (title, ach_level) in enumerate(scenarios):
#         # Modify ACh function for this scenario
#         ca3_pars['ACh'] = ach_level
#         wc_ca3 = WilsonCowan(**ca3_pars)
        
#         # Override ACh function for constant level
#         wc_ca3.ACh_func = lambda t: ach_level
        
#         # Simulate
#         rE, rI = wc_ca3.simulate(rE_init=0.32, rI_init=0.15)
        
#         # Plot
#         axes[i].plot(ca3_pars['range_t'], rE, 'b', label='Excitatory', linewidth=2)
#         axes[i].plot(ca3_pars['range_t'], rI, 'r', label='Inhibitory', linewidth=2)
#         axes[i].set_title(f'{title}')
#         axes[i].set_xlabel('Time (ms)')
#         axes[i].set_ylabel('Firing Rate')
#         axes[i].legend()
#         axes[i].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    print("=== Standard vs CA3 Model Comparison ===")
    wc_standard, wc_ca3, rE_std, rI_std, rE_ca3, rI_ca3 = compare_standard_vs_ca3()
    
    print("\n=== CA3 Activity with ACh Modulation ===")
    wc_ca3, rE_ca3, rI_ca3 = plot_ca3_with_ach()
    
    print("\n=== Comparing Different ACh Levels ===")
    # compare_ach_effects() 