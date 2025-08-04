import numpy as np
import matplotlib.pyplot as plt
from helpers import (set_parameters, set_ca3_parameters, find_multiple_fixed_points, 
                     get_eig_Jacobian, check_fp, simulate_wc, plot_complete_analysis)

def analyze_stability_simple(fp, pars):
    """
    Analyze stability of a fixed point using existing helper functions
    """
    try:
        eigenvalues = get_eig_Jacobian(fp, **pars)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # Determine stability
        if np.all(real_parts < -1e-10):
            stability = "stable"
        elif np.any(real_parts > 1e-10):
            stability = "unstable"
        else:
            stability = "marginally stable"
        
        # Determine type
        if np.all(np.abs(imag_parts) < 1e-10):
            if np.all(real_parts < -1e-10):
                fp_type = "stable node"
            elif np.any(real_parts > 1e-10):
                fp_type = "unstable node"
            else:
                fp_type = "saddle"
        else:
            if np.all(real_parts < -1e-10):
                fp_type = "stable focus"
            elif np.any(real_parts > 1e-10):
                fp_type = "unstable focus"
            else:
                fp_type = "center"
        
        return {
            'eigenvalues': eigenvalues,
            'stability': stability,
            'type': fp_type
        }
    except:
        return {
            'eigenvalues': np.array([0, 0]),
            'stability': "unknown",
            'type': "unknown"
        }

def classify_dynamical_regime_simple(pars):
    """
    Classify dynamical regime using existing helper functions
    """
    try:
        fixed_points = find_multiple_fixed_points(pars)
        
        if not fixed_points:
            return "no_fixed_points"
        
        stability_info = [analyze_stability_simple(fp, pars) for fp in fixed_points]
        
        # Count stable and unstable fixed points
        n_stable = sum(1 for s in stability_info if s['stability'] == 'stable')
        n_unstable = sum(1 for s in stability_info if s['stability'] == 'unstable')
        
        # Check for oscillations (Hopf bifurcation)
        has_oscillations = False
        for stability in stability_info:
            eigenvalues = stability['eigenvalues']
            real_parts = np.real(eigenvalues)
            imag_parts = np.imag(eigenvalues)
            
            # Check for complex conjugate eigenvalues with small real part
            if np.any(np.abs(real_parts) < 0.01) and np.any(np.abs(imag_parts) > 0.01):
                has_oscillations = True
                break
        
        # Classify regime
        if n_stable == 0 and n_unstable >= 1:
            return "unstable"
        elif n_stable == 1 and n_unstable == 0:
            return "stable_fixed_point"
        elif n_stable == 2 and n_unstable == 0:
            return "bistable"
        elif n_stable > 2 and n_unstable == 0:
            return "multistable"
        else:
            return "complex"
    except:
        return "error"

def create_2d_bifurcation_diagram_simple(param1_name, param1_range, param2_name, param2_range, 
                                        base_pars, regime_colors=None):
    """
    Create 2D bifurcation diagram using existing helper functions
    """
    if regime_colors is None:
        regime_colors = {
            'stable_fixed_point': 'green',
            'oscillatory': 'blue',
            'bistable': 'orange',
            'multistable': 'black',
            'unstable': 'red',
            'complex': 'purple',
            'no_fixed_points': 'gray',
            'error': 'white'
        }
    
    # Create parameter grid
    param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
    regime_grid = np.zeros_like(param1_grid, dtype=object)
    n_fixed_points_grid = np.zeros_like(param1_grid, dtype=int)
    
    print(f"Analyzing 2D parameter space: {param1_name} vs {param2_name}")
    print(f"Grid size: {len(param1_range)} x {len(param2_range)} = {len(param1_range) * len(param2_range)} points")
    
    # Analyze each point in parameter space
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            # Update parameters
            pars = base_pars.copy()
            pars[param1_name] = p1
            pars[param2_name] = p2
            
            # Classify dynamical regime
            regime = classify_dynamical_regime_simple(pars)
            regime_grid[j, i] = regime
            
            # Count fixed points
            try:
                fixed_points = find_multiple_fixed_points(pars)
                n_fixed_points_grid[j, i] = len(fixed_points)
            except:
                n_fixed_points_grid[j, i] = 0
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'2D Bifurcation Diagram: {param1_name} vs {param2_name}', fontsize=16)
    
    # Plot 1: Regime classification
    ax1 = axes[0, 0]
    unique_regimes = list(set(regime_grid.flatten()))
    
    for regime in unique_regimes:
        mask = (regime_grid == regime)
        # Use larger size and higher alpha for better visibility
        size = 40 if regime != 'stable_fixed_point' else 30
        alpha = 0.9 if regime != 'stable_fixed_point' else 0.7
        ax1.scatter(param1_grid[mask], param2_grid[mask], 
                   c=regime_colors.get(regime, 'gray'), 
                   label=regime, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_title('Dynamical Regimes')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of fixed points
    ax2 = axes[0, 1]
    im2 = ax2.contourf(param1_grid, param2_grid, n_fixed_points_grid, levels=10, cmap='viridis')
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_title('Number of Fixed Points')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Sample trajectories at selected points
    ax3 = axes[1, 0]
    
    # Select representative points from each regime
    selected_points = []
    for regime in unique_regimes:
        mask = (regime_grid == regime)
        if np.any(mask):
            # Find center point of this regime
            center_i = np.mean(np.where(mask)[1])
            center_j = np.mean(np.where(mask)[0])
            selected_points.append((param1_range[int(center_i)], param2_range[int(center_j)], regime))
    
    # Plot sample trajectories
    for p1, p2, regime in selected_points[:6]:  # Limit to 6 points for clarity
        pars = base_pars.copy()
        pars[param1_name] = p1
        pars[param2_name] = p2
        
        # Simulate trajectory
        try:
            rE, rI = simulate_wc(**pars)
            color = regime_colors.get(regime, 'gray')
            ax3.plot(rE, rI, color=color, alpha=0.7, linewidth=1, label=f'{regime}')
            ax3.plot(rE[0], rI[0], 'ko', markersize=3)
        except:
            continue
    
    ax3.set_xlabel('rE')
    ax3.set_ylabel('rI')
    ax3.set_title('Sample Trajectories')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fixed points at selected parameter values
    ax4 = axes[1, 1]
    
    # Show fixed points for a few selected parameter combinations
    selected_params = [
        (param1_range[len(param1_range)//4], param2_range[len(param2_range)//4]),
        (param1_range[len(param1_range)//2], param2_range[len(param2_range)//2]),
        (param1_range[3*len(param1_range)//4], param2_range[3*len(param2_range)//4])
    ]
    
    colors = ['blue', 'red', 'green']
    for (p1, p2), color in zip(selected_params, colors):
        pars = base_pars.copy()
        pars[param1_name] = p1
        pars[param2_name] = p2
        
        try:
            fixed_points = find_multiple_fixed_points(pars)
            for fp in fixed_points:
                stability = analyze_stability_simple(fp, pars)
                marker = 'o' if stability['stability'] == 'stable' else '^'
                ax4.plot(fp[0], fp[1], marker, color=color, markersize=8, 
                        label=f'{param1_name}={p1:.2f}, {param2_name}={p2:.2f}')
        except:
            continue
    
    ax4.set_xlabel('rE')
    ax4.set_ylabel('rI')
    ax4.set_title('Fixed Points at Selected Parameters')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return regime_grid, n_fixed_points_grid

def analyze_wEE_wIE_coexpression():
    """
    Analyze the coexpression of W_EE and W_IE parameters using existing helper functions
    """
    # Get base parameters
    std_pars = set_parameters()
    ca3_pars = set_ca3_parameters()
    
    print("=== 2D Bifurcation Analysis: W_EE vs W_IE ===")
    
    # Define parameter ranges (smaller grid for faster computation)
    wEE_range = np.linspace(2.0, 15.0, 15)  # Recurrent excitation
    wIE_range = np.linspace(2.0, 15.0, 15)  # Excitatory to inhibitory

    A_E_range = np.linspace(0.1, 2.5, 15)
    A_I_range = np.linspace(0.1, 2.5, 15)
    
    # Analyze standard parameters
    # print("\nAnalyzing Standard Wilson-Cowan Model...")
    # std_regime_grid, std_n_fps = create_2d_bifurcation_diagram_simple(
    #     'wEE', wEE_range, 'wIE', wIE_range, std_pars
    # )
    
    # Analyze CA3 parameters
    print("\nAnalyzing CA3 Wilson-Cowan Model...")
    ca3_regime_grid, ca3_n_fps = create_2d_bifurcation_diagram_simple(
        'wEE', wEE_range, 'wIE', wIE_range, ca3_pars
    )

    # Analyze CA3 parameters
    print("\nAnalyzing CA3 Wilson-Cowan Model...")
    ca3_regime_grid, ca3_n_fps = create_2d_bifurcation_diagram_simple(
        'WEE', wEE_range, 'A_E', A_E_range, ca3_pars
    )

    print("\nAnalyzing CA3 Wilson-Cowan Model...")
    ca3_regime_grid, ca3_n_fps = create_2d_bifurcation_diagram_simple(
        'A_E', A_E_range, 'A_I', A_I_range, ca3_pars
    )
    
    # Create comparison plot
    create_comparison_plot_simple(wEE_range, wIE_range, std_regime_grid, ca3_regime_grid)
    
    return {
        'std_regime_grid': std_regime_grid,
        'ca3_regime_grid': ca3_regime_grid,
        'std_n_fps': std_n_fps,
        'ca3_n_fps': ca3_n_fps
    }

def create_comparison_plot_simple(wEE_range, wIE_range, std_regime_grid, ca3_regime_grid):
    """
    Create comparison plot between standard and CA3 models
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison: Standard vs CA3 Wilson-Cowan Models', fontsize=16)
    
    # Define regime colors
    regime_colors = {
        'stable_fixed_point': 'green',
        'oscillatory': 'blue',
        'bistable': 'orange',
        'multistable': 'red',
        'unstable': 'red',
        'complex': 'purple',
        'no_fixed_points': 'gray',
        'error': 'white'
    }
    
    # Create parameter grids
    wEE_grid, wIE_grid = np.meshgrid(wEE_range, wIE_range)
    
    # Plot 1: Standard model regimes
    ax1 = axes[0, 0]
    unique_regimes = list(set(std_regime_grid.flatten()))
    for regime in unique_regimes:
        mask = (std_regime_grid == regime)
        # Use larger size and higher alpha for better visibility
        size = 40 if regime != 'stable_fixed_point' else 30
        alpha = 0.9 if regime != 'stable_fixed_point' else 0.7
        ax1.scatter(wEE_grid[mask], wIE_grid[mask], 
                   c=regime_colors.get(regime, 'gray'), 
                   label=regime, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('W_EE')
    ax1.set_ylabel('W_IE')
    ax1.set_title('Standard Model: Dynamical Regimes')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CA3 model regimes
    ax2 = axes[0, 1]
    unique_regimes_ca3 = list(set(ca3_regime_grid.flatten()))
    for regime in unique_regimes_ca3:
        mask = (ca3_regime_grid == regime)
        # Use larger size and higher alpha for better visibility
        size = 40 if regime != 'stable_fixed_point' else 30
        alpha = 0.9 if regime != 'stable_fixed_point' else 0.7
        ax2.scatter(wEE_grid[mask], wIE_grid[mask], 
                   c=regime_colors.get(regime, 'gray'), 
                   label=regime, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('W_EE')
    ax2.set_ylabel('W_IE')
    ax2.set_title('CA3 Model: Dynamical Regimes')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime differences
    ax3 = axes[1, 0]
    regime_differences = np.zeros_like(wEE_grid, dtype=int)
    
    for i in range(len(wEE_range)):
        for j in range(len(wIE_range)):
            if std_regime_grid[j, i] != ca3_regime_grid[j, i]:
                regime_differences[j, i] = 1
    
    if np.any(regime_differences):
        ax3.scatter(wEE_grid[regime_differences == 1], wIE_grid[regime_differences == 1], 
                   c='red', s=50, alpha=0.9, label='Different regimes', edgecolors='black', linewidth=0.5)
    
    ax3.scatter(wEE_grid[regime_differences == 0], wIE_grid[regime_differences == 0], 
               c='green', s=30, alpha=0.7, label='Same regime', edgecolors='black', linewidth=0.5)
    
    ax3.set_xlabel('W_EE')
    ax3.set_ylabel('W_IE')
    ax3.set_title('Regime Differences (Standard vs CA3)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter space summary
    ax4 = axes[1, 1]
    
    # Count regimes in each model
    std_regime_counts = {}
    ca3_regime_counts = {}
    
    for regime in set(std_regime_grid.flatten()):
        std_regime_counts[regime] = np.sum(std_regime_grid == regime)
    
    for regime in set(ca3_regime_grid.flatten()):
        ca3_regime_counts[regime] = np.sum(ca3_regime_grid == regime)
    
    # Create bar plot
    all_regimes = list(set(list(std_regime_counts.keys()) + list(ca3_regime_counts.keys())))
    x = np.arange(len(all_regimes))
    width = 0.35
    
    std_counts = [std_regime_counts.get(regime, 0) for regime in all_regimes]
    ca3_counts = [ca3_regime_counts.get(regime, 0) for regime in all_regimes]
    
    ax4.bar(x - width/2, std_counts, width, label='Standard', alpha=0.8)
    ax4.bar(x + width/2, ca3_counts, width, label='CA3', alpha=0.8)
    
    ax4.set_xlabel('Dynamical Regimes')
    ax4.set_ylabel('Number of Parameter Combinations')
    ax4.set_title('Regime Distribution')
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_regimes, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_current_parameters():
    """
    Analyze the current parameter sets to understand their stability properties
    """
    print("=== Current Parameter Analysis ===")
    
    # Get current parameters
    std_pars = set_parameters()
    ca3_pars = set_ca3_parameters()
    
    print("\nStandard Parameters:")
    for key, value in std_pars.items():
        if key not in ['range_t']:  # Skip the time array
            print(f"  {key}: {value}")
    
    print("\nCA3 Parameters:")
    for key, value in ca3_pars.items():
        if key not in ['range_t']:  # Skip the time array
            print(f"  {key}: {value}")
    
    # Find fixed points for both models
    print("\n=== Fixed Point Analysis ===")
    
    # Standard model
    print("\nStandard Model Fixed Points:")
    std_fps = find_multiple_fixed_points(std_pars)
    for i, fp in enumerate(std_fps):
        stability = analyze_stability_simple(fp, std_pars)
        print(f"  FP {i+1}: rE={fp[0]:.4f}, rI={fp[1]:.4f}")
        print(f"    Stability: {stability['stability']}, Type: {stability['type']}")
        print(f"    Eigenvalues: {stability['eigenvalues']}")
    
    # CA3 model
    print("\nCA3 Model Fixed Points:")
    ca3_fps = find_multiple_fixed_points(ca3_pars)
    for i, fp in enumerate(ca3_fps):
        stability = analyze_stability_simple(fp, ca3_pars)
        print(f"  FP {i+1}: rE={fp[0]:.4f}, rI={fp[1]:.4f}")
        print(f"    Stability: {stability['stability']}, Type: {stability['type']}")
        print(f"    Eigenvalues: {stability['eigenvalues']}")
    
    # Classify dynamical regimes
    print("\n=== Dynamical Regime Classification ===")
    std_regime = classify_dynamical_regime_simple(std_pars)
    ca3_regime = classify_dynamical_regime_simple(ca3_pars)
    
    print(f"Standard Model: {std_regime}")
    print(f"CA3 Model: {ca3_regime}")
    
    return {
        'std_pars': std_pars,
        'ca3_pars': ca3_pars,
        'std_fps': std_fps,
        'ca3_fps': ca3_fps,
        'std_regime': std_regime,
        'ca3_regime': ca3_regime
    }

if __name__ == "__main__":
    # First analyze current parameters
    current_analysis = analyze_current_parameters()
    
    # Then run 2D bifurcation analysis
    results = analyze_wEE_wIE_coexpression()
    
    print("\n=== Analysis Complete ===")
    print("Key findings:")
    print("1. Current parameter sets analyzed for stability")
    print("2. 2D bifurcation diagrams created for W_EE vs W_IE")
    print("3. Comparison between standard and CA3 models") 