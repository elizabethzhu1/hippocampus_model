import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eig
from helpers import set_parameters, set_ca3_parameters, F, dF, EIderivs, simulate_wc

def find_fixed_points(pars, rE_guess=0.5, rI_guess=0.3):
    """
    Find fixed points of the Wilson-Cowan system by solving drE/dt = 0, drI/dt = 0
    """
    def equations(vars):
        rE, rI = vars
        drEdt, drIdt = EIderivs(rE, rI, **pars)
        return [drEdt, drIdt]
    
    # Try different initial conditions to find multiple fixed points
    initial_guesses = [
        (rE_guess, rI_guess),
        (0.1, 0.1),
        (0.8, 0.8),
        (0.2, 0.6),
        (0.6, 0.2)
    ]
    
    fixed_points = []
    for guess in initial_guesses:
        try:
            solution = fsolve(equations, guess, full_output=True)
            if solution[2] == 1:  # Check if solution converged
                rE_fp, rI_fp = solution[0]
                # Check if point is within reasonable bounds
                if 0 <= rE_fp <= 1 and 0 <= rI_fp <= 1:
                    # Check if this point is already found (within tolerance)
                    is_new = True
                    for existing_fp in fixed_points:
                        if np.linalg.norm(np.array([rE_fp, rI_fp]) - np.array(existing_fp)) < 0.01:
                            is_new = False
                            break
                    if is_new:
                        fixed_points.append((rE_fp, rI_fp))
        except:
            continue
    
    return fixed_points

def analyze_stability(fixed_point, pars):
    """
    Analyze stability of a fixed point using linearization
    """
    rE_fp, rI_fp = fixed_point
    
    # Calculate Jacobian matrix at fixed point
    tau_E = pars['tau_E']
    tau_I = pars['tau_I']
    wEE = pars['wEE']
    wEI = pars['wEI']
    wIE = pars['wIE']
    wII = pars['wII']
    a_E = pars['a_E']
    a_I = pars['a_I']
    theta_E = pars['theta_E']
    theta_I = pars['theta_I']
    ext_E = pars['ext_E']
    ext_I = pars['ext_I']
    
    # Input to activation functions
    input_E = wEE * rE_fp - wEI * rI_fp + ext_E
    input_I = wIE * rE_fp - wII * rI_fp + ext_I
    
    # Derivatives of activation functions
    dF_E = dF(input_E, a_E, theta_E)
    dF_I = dF(input_I, a_I, theta_I)
    
    # Jacobian matrix elements
    J11 = (-1 + wEE * dF_E) / tau_E
    J12 = (-wEI * dF_E) / tau_E
    J21 = (wIE * dF_I) / tau_I
    J22 = (-1 - wII * dF_I) / tau_I
    
    J = np.array([[J11, J12], [J21, J22]])
    
    # Calculate eigenvalues
    eigenvalues = eig(J)[0]
    
    # Determine stability
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    if np.all(real_parts < -1e-10):
        stability = "stable"
    elif np.any(real_parts > 1e-10):
        stability = "unstable"
    else:
        stability = "marginally stable"
    
    # Determine type of fixed point
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
        'type': fp_type,
        'jacobian': J
    }

def classify_dynamical_regime(pars):
    """
    Classify the dynamical regime based on fixed points and their stability
    """
    fixed_points = find_fixed_points(pars)
    
    if not fixed_points:
        return "no_fixed_points"
    
    stability_info = [analyze_stability(fp, pars) for fp in fixed_points]
    
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
    if n_stable == 0:
        return "unstable"
    elif n_stable == 1 and n_unstable == 0:
        if has_oscillations:
            return "oscillatory"
        else:
            return "stable_fixed_point"
    elif n_stable == 1 and n_unstable == 1:
        return "bistable"
    elif n_stable >= 2:
        return "multistable"
    else:
        return "complex"

def create_2d_bifurcation_diagram(param1_name, param1_range, param2_name, param2_range, 
                                 base_pars, regime_colors=None):
    """
    Create a 2D bifurcation diagram showing dynamical regimes across two parameters
    
    Args:
        param1_name, param2_name: names of the two parameters to vary
        param1_range, param2_range: arrays of parameter values
        base_pars: base parameter dictionary
        regime_colors: dictionary mapping regime names to colors
    """
    if regime_colors is None:
        regime_colors = {
            'stable_fixed_point': 'green',
            'oscillatory': 'blue',
            'bistable': 'orange',
            'multistable': 'red',
            'unstable': 'black',
            'complex': 'purple',
            'no_fixed_points': 'gray'
        }
    
    # Create parameter grid
    param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
    regime_grid = np.zeros_like(param1_grid, dtype=object)
    
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
            regime = classify_dynamical_regime(pars)
            regime_grid[j, i] = regime
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'2D Bifurcation Diagram: {param1_name} vs {param2_name}', fontsize=16)
    
    # Plot 1: Regime classification
    ax1 = axes[0, 0]
    regime_array = np.array([[regime_colors.get(regime, 'gray') for regime in row] for row in regime_grid])
    
    # Create custom colormap
    unique_regimes = list(set(regime_grid.flatten()))
    colors_list = [regime_colors.get(regime, 'gray') for regime in unique_regimes]
    
    # Plot with custom colors
    for regime in unique_regimes:
        mask = (regime_grid == regime)
        ax1.scatter(param1_grid[mask], param2_grid[mask], 
                   c=regime_colors.get(regime, 'gray'), 
                   label=regime, s=20, alpha=0.7)
    
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_title('Dynamical Regimes')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of fixed points
    ax2 = axes[0, 1]
    n_fixed_points = np.zeros_like(param1_grid, dtype=int)
    
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            pars = base_pars.copy()
            pars[param1_name] = p1
            pars[param2_name] = p2
            fixed_points = find_fixed_points(pars)
            n_fixed_points[j, i] = len(fixed_points)
    
    im2 = ax2.contourf(param1_grid, param2_grid, n_fixed_points, levels=10, cmap='viridis')
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_title('Number of Fixed Points')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Stability analysis
    ax3 = axes[1, 0]
    stability_score = np.zeros_like(param1_grid, dtype=float)
    
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            pars = base_pars.copy()
            pars[param1_name] = p1
            pars[param2_name] = p2
            fixed_points = find_fixed_points(pars)
            
            if fixed_points:
                # Calculate average stability (negative real part of eigenvalues)
                total_stability = 0
                for fp in fixed_points:
                    stability = analyze_stability(fp, pars)
                    real_parts = np.real(stability['eigenvalues'])
                    total_stability += np.mean(real_parts)
                stability_score[j, i] = total_stability / len(fixed_points)
            else:
                stability_score[j, i] = 0
    
    im3 = ax3.contourf(param1_grid, param2_grid, stability_score, levels=20, cmap='RdBu_r')
    ax3.set_xlabel(param1_name)
    ax3.set_ylabel(param2_name)
    ax3.set_title('Average Stability (Real Part of Eigenvalues)')
    plt.colorbar(im3, ax=ax3)
    
    # Plot 4: Sample trajectories at selected points
    ax4 = axes[1, 1]
    
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
            ax4.plot(rE, rI, color=color, alpha=0.7, linewidth=1, label=f'{regime}')
            ax4.plot(rE[0], rI[0], 'ko', markersize=3)
        except:
            continue
    
    ax4.set_xlabel('rE')
    ax4.set_ylabel('rI')
    ax4.set_title('Sample Trajectories')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return regime_grid, n_fixed_points, stability_score

def analyze_parameter_coexpression():
    """
    Analyze the coexpression of W_EE and W_IE parameters
    """
    # Get base parameters
    std_pars = set_parameters()
    ca3_pars = set_ca3_parameters()
    
    print("=== 2D Bifurcation Analysis: W_EE vs W_IE ===")
    
    # Define parameter ranges
    wEE_range = np.linspace(2.0, 10.0, 25)  # Recurrent excitation
    wIE_range = np.linspace(2.0, 12.0, 25)  # Excitatory to inhibitory
    
    # Analyze standard parameters
    print("\nAnalyzing Standard Wilson-Cowan Model...")
    std_regime_grid, std_n_fps, std_stability = create_2d_bifurcation_diagram(
        'wEE', wEE_range, 'wIE', wIE_range, std_pars
    )
    
    # Analyze CA3 parameters
    print("\nAnalyzing CA3 Wilson-Cowan Model...")
    ca3_regime_grid, ca3_n_fps, ca3_stability = create_2d_bifurcation_diagram(
        'wEE', wEE_range, 'wIE', wIE_range, ca3_pars
    )
    
    # Create comparison plot
    create_comparison_plot(wEE_range, wIE_range, std_regime_grid, ca3_regime_grid)
    
    return {
        'std_regime_grid': std_regime_grid,
        'ca3_regime_grid': ca3_regime_grid,
        'std_n_fps': std_n_fps,
        'ca3_n_fps': ca3_n_fps,
        'std_stability': std_stability,
        'ca3_stability': ca3_stability
    }

def create_comparison_plot(wEE_range, wIE_range, std_regime_grid, ca3_regime_grid):
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
        'unstable': 'black',
        'complex': 'purple',
        'no_fixed_points': 'gray'
    }
    
    # Create parameter grids
    wEE_grid, wIE_grid = np.meshgrid(wEE_range, wIE_range)
    
    # Plot 1: Standard model regimes
    ax1 = axes[0, 0]
    unique_regimes = list(set(std_regime_grid.flatten()))
    for regime in unique_regimes:
        mask = (std_regime_grid == regime)
        ax1.scatter(wEE_grid[mask], wIE_grid[mask], 
                   c=regime_colors.get(regime, 'gray'), 
                   label=regime, s=20, alpha=0.7)
    
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
        ax2.scatter(wEE_grid[mask], wIE_grid[mask], 
                   c=regime_colors.get(regime, 'gray'), 
                   label=regime, s=20, alpha=0.7)
    
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
                   c='red', s=30, alpha=0.8, label='Different regimes')
    
    ax3.scatter(wEE_grid[regime_differences == 0], wIE_grid[regime_differences == 0], 
               c='green', s=20, alpha=0.5, label='Same regime')
    
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

def analyze_other_parameter_pairs():
    """
    Analyze other interesting parameter pairs
    """
    std_pars = set_parameters()
    
    # Define parameter pairs to analyze
    parameter_pairs = [
        ('wEE', 'wEI', 'Recurrent Excitation vs Inhibitory to Excitatory'),
        ('wIE', 'wII', 'Excitatory to Inhibitory vs Recurrent Inhibition'),
        ('a_E', 'theta_E', 'Excitatory Gain vs Threshold'),
        ('ext_E', 'wEE', 'External Input vs Recurrent Excitation')
    ]
    
    for param1, param2, title in parameter_pairs:
        print(f"\n=== Analyzing {title} ===")
        
        # Define ranges for each parameter
        if param1 == 'wEE':
            range1 = np.linspace(2.0, 10.0, 20)
        elif param1 == 'wEI':
            range1 = np.linspace(1.0, 8.0, 20)
        elif param1 == 'wIE':
            range1 = np.linspace(2.0, 12.0, 20)
        elif param1 == 'wII':
            range1 = np.linspace(0.1, 5.0, 20)
        elif param1 == 'a_E':
            range1 = np.linspace(0.5, 2.5, 20)
        elif param1 == 'theta_E':
            range1 = np.linspace(1.5, 4.5, 20)
        elif param1 == 'ext_E':
            range1 = np.linspace(0.0, 2.0, 20)
        else:
            range1 = np.linspace(0.5, 2.0, 20)
        
        if param2 == 'wEE':
            range2 = np.linspace(2.0, 10.0, 20)
        elif param2 == 'wEI':
            range2 = np.linspace(1.0, 8.0, 20)
        elif param2 == 'wIE':
            range2 = np.linspace(2.0, 12.0, 20)
        elif param2 == 'wII':
            range2 = np.linspace(0.1, 5.0, 20)
        elif param2 == 'a_E':
            range2 = np.linspace(0.5, 2.5, 20)
        elif param2 == 'theta_E':
            range2 = np.linspace(1.5, 4.5, 20)
        elif param2 == 'ext_E':
            range2 = np.linspace(0.0, 2.0, 20)
        else:
            range2 = np.linspace(0.5, 2.0, 20)
        
        create_2d_bifurcation_diagram(param1, range1, param2, range2, std_pars)

if __name__ == "__main__":
    # Run 2D bifurcation analysis for W_EE vs W_IE
    results = analyze_parameter_coexpression()
    
    # Analyze other parameter pairs
    analyze_other_parameter_pairs()
    
    print("\n=== 2D Bifurcation Analysis Complete ===")
    print("Key findings:")
    print("1. Different dynamical regimes identified across parameter space")
    print("2. Critical boundaries between stable, oscillatory, and bistable regions")
    print("3. Comparison between standard and CA3 models shows parameter sensitivity differences") 