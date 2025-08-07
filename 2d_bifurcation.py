import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eig
from helpers import set_ca3_parameters, dF, EIderivs, simulate_wc
import argparse

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
            # solve for zero derivatives
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
            print("No solution found for initial guess", guess)
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
    
    # Handle ext_E and ext_I - use mean if they're arrays
    ext_E = pars['ext_E']
    ext_I = pars['ext_I']
    
    if hasattr(ext_E, '__len__') and len(ext_E) > 1:
        ext_E = np.mean(ext_E)
    if hasattr(ext_I, '__len__') and len(ext_I) > 1:
        ext_I = np.mean(ext_I)
    
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
    # else:
    #     stability = "marginally stable"

    oscillatory = False
    if np.any(imag_parts > 1e-10):
        oscillatory = True 
    
    saddle = False
    if np.any(real_parts > -1e-10) and np.any(real_parts > 1e-10):
        saddle = True
    
    return {
        'eigenvalues': eigenvalues,
        'stability': stability,
        'oscillatory': oscillatory,
        'saddle': saddle,
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
    oscillatory = stability_info[0]['oscillatory']
    saddle = stability_info[0]['saddle']
    
    # Classify regime
    if n_stable == 0 and n_unstable > 0:
        if oscillatory:
            return "unstable (oscillatory)"
        elif saddle:
            return "unstable (saddle)"
    # One stable fixed point
    elif n_stable == 1 and n_unstable == 0:
        
        if oscillatory:
            return "stable (oscillatory)"
        elif saddle:
            return "stable (saddle)"
    elif n_stable == 1 and n_unstable == 1:
        return "bistable"
    elif n_stable >= 2:
        return "multistable"


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
            'bistable': 'orange',
            'multistable': 'purple',
            'unstable (oscillatory)': 'black',
            'stable (oscillatory)': 'blue',
            'unstable (saddle)': 'red',
            'stable (saddle)': 'orange',
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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

def analyze_parameter_coexpression(args):
    """
    Analyze the coexpression of two parameters
    """

    ca3_pars = set_ca3_parameters()
    ca3_pars['ext_E'] = args.ext_E
    ca3_pars['ext_I'] = args.ext_I

    param1 = args.param1
    param2 = args.param2
    
    print(f"=== 2D Bifurcation Analysis: {param1} vs {param2} ===")
    
    # Define parameter ranges
    p1_range = np.linspace(args.min1, args.max1, 25)  # Recurrent excitation
    p2_range = np.linspace(args.min2, args.max2, 25)  # Excitatory to inhibitory
    
    # Analyze CA3 parameters
    print("\nAnalyzing CA3 Wilson-Cowan Model...")
    ca3_regime_grid, ca3_n_fps, ca3_stability = create_2d_bifurcation_diagram(
        param1, p1_range, param2, p2_range, ca3_pars
    )
    

    return {
        'ca3_regime_grid': ca3_regime_grid,
        'ca3_n_fps': ca3_n_fps,
        'ca3_stability': ca3_stability
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bifurcation Analysis')
    parser.add_argument('--region', type=str, required=True, choices=['ca3', 'ca1', 'dg'], help='Specify region to analyze')
    parser.add_argument('--param1', type=str, required=True, choices=['wEE', 'wIE', 'wII', 'wEI', 'a_E', 'theta_E', 'a_I', 'theta_I'], default='wEE', help='Specify first parameter')
    parser.add_argument('--param2', type=str, required=True, choices=['wEE', 'wIE', 'wII', 'wEI', 'a_E', 'theta_E', 'a_I', 'theta_I'], default='wIE', help='Specify second parameter')

    # Optional parameters
    parser.add_argument('--min1', type=float, required=False, default=2.0, help='Specify min value for the first parameter')
    parser.add_argument('--max1', type=float, required=False, default=10.0, help='Specify max value for the first parameter')
    parser.add_argument('--min2', type=float, required=False, default=2.0, help='Specify min value for the second parameter')
    parser.add_argument('--max2', type=float, required=False, default=10.0, help='Specify max value for the second parameter')
    parser.add_argument('--ext_E', type=float, required=False, default=0.5, help='Specify ext_E')
    parser.add_argument('--ext_I', type=float, required=False, default=0.5, help='Specify ext_I')

    args = parser.parse_args()

    # Run 2D bifurcation analysis for choice of parameters 
    results = analyze_parameter_coexpression(args)
