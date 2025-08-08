import numpy as np
import matplotlib.pyplot as plt
from models.ca3 import CA3_WilsonCowan
from helpers import set_ca3_parameters

def plot_ca3_with_ach():
    """
    Plot CA3 excitatory-inhibitory activity with ACh modulation
    """
    # Get CA3 parameters
    ca3_pars = set_ca3_parameters()

    # Create CA3 Wilson-Cowan model with DG input + without (just noise)
    ca3_pars['is_acetylcholine'] = True
    ca3_pars['is_DG_input'] = False
    wc_ca3 = CA3_WilsonCowan(**ca3_pars)

    ca3_pars['is_acetylcholine'] = True
    ca3_pars['is_DG_input'] = True
    wc_ca3_dg = CA3_WilsonCowan(**ca3_pars)
    
    # Simulate both models
    rE_ca3, rI_ca3 = wc_ca3.simulate(rE_init=0.32, rI_init=0.15)
    rE_ca3_dg, rI_ca3_dg = wc_ca3_dg.simulate(rE_init=0.32, rI_init=0.15)
    
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    
    # Calculate ACh values
    ach_values = [wc_ca3.ACh_func(t) for t in range(len(wc_ca3.range_t))]
    
    # Plot neural activity without DG
    ax1.plot(wc_ca3.range_t, rE_ca3, 'b', label='E', linewidth=2)
    ax1.plot(wc_ca3.range_t, rI_ca3, 'r', label='I', linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(wc_ca3.range_t, ach_values, '#00FF00', label='ACh', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Firing Rate')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('CA3 Activity (no DG)')
    ax1_twin.set_ylabel('ACh Level', color='#00FF00')
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot neural activity with DG
    ax2.plot(wc_ca3.range_t, rE_ca3_dg, 'b', label='E', linewidth=2)
    ax2.plot(wc_ca3.range_t, rI_ca3_dg, 'r', label='I', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(wc_ca3.range_t, ach_values, '#00FF00', label='ACh', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Firing Rate')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('CA3 Activity (with DG)')
    ax2_twin.set_ylabel('ACh Level', color='#00FF00')
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Sync y-axis limits for activity plots
    ymin = min(min(rE_ca3), min(rI_ca3), min(rE_ca3_dg), min(rI_ca3_dg))
    ymax = max(max(rE_ca3), max(rI_ca3), max(rE_ca3_dg), max(rI_ca3_dg))
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    # Sync ACh axis limits
    ax1_twin.set_ylim(0, 1)
    ax2_twin.set_ylim(0, 1)
    
    plt.tight_layout()

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


if __name__ == "__main__":
    # print("=== Standard vs CA3 Model Comparison ===")
    # wc_standard, wc_ca3, rE_std, rI_std, rE_ca3, rI_ca3 = compare_standard_vs_ca3()
    
    print("\n=== CA3 Activity with ACh Modulation ===")
    wc_ca3, rE_ca3, rI_ca3 = plot_ca3_with_ach()
