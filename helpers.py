import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def set_parameters(**kwargs):
  pars = {}

  # Excitatory parameters
  pars['tau_E'] = 1.     # Timescale of the E population [ms]
  pars['a_E'] = 1.2      # Gain of the E population
  pars['theta_E'] = 2.8  # Threshold of the E population

  # Inhibitory parameters
  pars['tau_I'] = 2.    # Timescale of the I population [ms]
  pars['a_I'] = 0.8     # Gain of the I population
  pars['theta_I'] = 4.  # Threshold of the I population

  # Produce oscillations?
  pars['wEE'] = 4.8  # recurrent excitation
  pars['wEI'] = 4.8  # inhibitory to excitatory

  pars['wII'] = 1.2  # recurrent inhibition
  pars['wIE'] = 6   # excitatory to inhibitory

  # External input
  pars['ext_E'] = 0.8
  pars['ext_I'] = 0

  pars['tau_A'] = 0.1

  # simulation parameters
  pars['T'] = 1500        # Total duration of simulation [ms]
  pars['dt'] = .1       # Simulation time step [ms]
  pars['rE_init'] = 0.2  # Initial value of E
  pars['rI_init'] = 0.2  # Initial value of I

  # External parameters if any
  for k in kwargs:
      pars[k] = kwargs[k]

  # Vector of discretized time points [ms]
  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

  return pars

def set_ca3_parameters(**kwargs):
  pars = {}

  # based on CA3 paper

  pars['tau_E'] = 1.8    # Timescale of the E population [ms]
  pars['tau_I'] = 4.8    # Timescale of the I population [ms]

  pars['a_E'] = 1.0     # Gain of the E population
  pars['a_I'] = 2.0    # Gain of the I population
  
  pars['theta_E'] = 2.8  # Threshold of the E population
  pars['theta_I'] = 3.0  # Threshold of the I population

  pars['wEE'] = 10 # E to E -- big bifurcation between 6.8 and 6.9, all else constant
  pars['wIE'] = 12  # E to I
  
  pars['wEI'] = 10  # I to E
  pars['wII'] = 10  # I to I

  # External input
  pars['ext_E'] = 0.5
  pars['ext_I'] = 0  # interneurons are being locally excited not externally

  pars['tau_A'] = 10  # not used for now

  # simulation parameters
  pars['T'] = 1500.        # Total duration of simulation [ms]
  pars['dt'] = .1       # Simulation time step [ms]
  pars['rE_init'] = 0.2  # Initial value of E
  pars['rI_init'] = 0.2  # Initial value of I

  # External parameters if any
  for k in kwargs:
      pars[k] = kwargs[k]

  # Vector of discretized time points [ms]
  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

  return pars

def set_ca1_parameters(**kwargs):
  pars = {}
  
  # Time constants - CA1 pyramidal cells have faster dynamics than CA3
  pars['tau_E'] = 1.6
  pars['tau_I'] = 4.0
  
  # Gain parameters
  pars['a_E'] = 1.0   
  pars['a_I'] = 2.0   
  
  # Threshold parameters
  pars['theta_E'] = 2.8  
  pars['theta_I'] = 3.0 
  
  # Connection weights - CA1 has different connectivity patterns
  pars['wEE'] = 8     
  pars['wIE'] = 8    
  pars['wEI'] = 5   
  pars['wII'] = 5      
  
  # External input - CA1 receives different input patterns
  pars['ext_E'] = 0.5   
  pars['ext_I'] = 0    
  
  pars['tau_A'] = 10
  
  # Simulation parameters
  pars['T'] = 1500.      
  pars['dt'] = .1   
  pars['rE_init'] = 0.2 
  pars['rI_init'] = 0.2 
  
  # External parameters if any
  for k in kwargs:
      pars[k] = kwargs[k]
  
  # Vector of discretized time points [ms]
  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])
  
  return pars

def F(x, a, theta):
  """
  Population activation function, F-I curve

  Args:
    x     : the population input
    a     : the gain of the function
    theta : the threshold of the function

  Returns:
    f     : the population activation response f(x) for input x
  """

  # add the expression of f = F(x)
  f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1
  # f = (1 + np.exp(-a)) ** -1

  return f

# def F(x, a, theta):
#   """
#   Population activation function, F-I curve

#   Args:
#     x     : the population input
#     a     : the gain of the function
#     theta : the threshold of the function

#   Returns:
#     f     : the population activation response f(x) for input x
#   """

#   # add the expression of f = F(x)
#   f = (1 + np.exp(-a * (x - theta)))**-1

#   return f


def dF(x, a, theta):
  """
  Derivative of the population activation function.

  Args:
    x     : the population input
    a     : the gain of the function
    theta : the threshold of the function

  Returns:
    dFdx  :  Derivative of the population activation function.
  """

  dFdx = a * np.exp(-a * (x - theta)) * (1 + np.exp(-a * (x - theta)))**-2

  return dFdx


def F_inv(x, a, theta):
  """
  Args:
    x         : the population input
    a         : the gain of the function
    theta     : the threshold of the function

  Returns:
    F_inverse : value of the inverse function
  """

  # Calculate Finverse (ln(x) can be calculated as np.log(x))
  F_inverse = -1/a * np.log((x + (1 + np.exp(a * theta))**-1)**-1 - 1) + theta

  return F_inverse


### Plotting Functions Below ###
# @title Plotting Functions

def plot_FI_inverse(x, a, theta):
  f, ax = plt.subplots()
  ax.plot(x, F_inv(x, a=a, theta=theta))
  ax.set(xlabel="$x$", ylabel="$F^{-1}(x)$")


def plot_FI_EI(x, FI_exc, FI_inh):
  plt.figure()
  plt.plot(x, FI_exc, 'b', label='E population')
  plt.plot(x, FI_inh, 'r', label='I population')
  plt.legend(loc='lower right')
  plt.xlabel('x (a.u.)')
  plt.ylabel('F(x)')
  plt.show()


def my_test_plot(t, rE1_std, rI1_std, rE2_std, rI2_std, rE1_ca3, rI1_ca3, rE2_ca3, rI2_ca3):

  plt.figure(figsize=(10, 8))
  
  # Standard WC model plots
  ax1 = plt.subplot(221)
  pars = set_parameters()

  ax1.plot(pars['range_t'], rE1_std, 'b', label='E population')
  ax1.plot(pars['range_t'], rI1_std, 'r', label='I population')
  ax1.set_ylabel('Activity')
  ax1.set_title('Standard WC - Initial Condition 1')
  ax1.legend(loc='best')

  ax2 = plt.subplot(222, sharey=ax1)
  ax2.plot(pars['range_t'], rE2_std, 'b', label='E population')
  ax2.plot(pars['range_t'], rI2_std, 'r', label='I population')
  ax2.set_title('Standard WC - Initial Condition 2')
  ax2.legend(loc='best')

  # CA3 WC model plots  
  ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
  ax3.plot(pars['range_t'], rE1_ca3, 'b', label='E population')
  ax3.plot(pars['range_t'], rI1_ca3, 'r', label='I population')
  ax3.set_xlabel('t (ms)')
  ax3.set_ylabel('Activity')
  ax3.set_title('CA3 WC - Initial Condition 1')
  ax3.legend(loc='best')

  ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)
  ax4.plot(pars['range_t'], rE2_ca3, 'b', label='E population')
  ax4.plot(pars['range_t'], rI2_ca3, 'r', label='I population')
  ax4.set_xlabel('t (ms)')
  ax4.set_title('CA3 WC - Initial Condition 2')
  ax4.legend(loc='best')

  plt.tight_layout()
  # plt.show() # removed to allow all plots to be shown at once


def my_test_plot_ca3_to_ca1(t, rE1_ca3, rI1_ca3, rE2_ca3, rI2_ca3, rE1_ca1, rI1_ca1, rE2_ca1, rI2_ca1):

  plt.figure(figsize=(10, 8))
  
  # Standard WC model plots
  ax1 = plt.subplot(221)
  pars = set_parameters()

  ax1.plot(pars['range_t'], rE1_ca3, 'b', label='E population')
  ax1.plot(pars['range_t'], rI1_ca3, 'r', label='I population')
  ax1.set_ylabel('Activity')
  ax1.set_title('CA3 WC - Initial Condition 1')
  ax1.legend(loc='best')

  ax2 = plt.subplot(222, sharey=ax1)
  ax2.plot(pars['range_t'], rE2_ca3, 'b', label='E population')
  ax2.plot(pars['range_t'], rI2_ca3, 'r', label='I population')
  ax2.set_title('CA3 WC - Initial Condition 2')
  ax2.legend(loc='best')

  # CA3 WC model plots  
  ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
  ax3.plot(pars['range_t'], rE1_ca1, 'b', label='E population')
  ax3.plot(pars['range_t'], rI1_ca1, 'r', label='I population')
  ax3.set_xlabel('t (ms)')  
  ax3.set_ylabel('Activity')
  ax3.set_title('CA1 WC - Initial Condition 1')
  ax3.legend(loc='best')

  ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)
  ax4.plot(pars['range_t'], rE2_ca1, 'b', label='E population')
  ax4.plot(pars['range_t'], rI2_ca1, 'r', label='I population')
  ax4.set_xlabel('t (ms)')
  ax4.set_title('CA1 WC - Initial Condition 2')
  ax4.legend(loc='best')

  plt.tight_layout()
  plt.show() # removed to allow all plots to be shown at once
def my_test_plot_ca3_to_ca1_with_dg(t, rE1_ca3, rI1_ca3, rE2_ca3, rI2_ca3, rE1_ca1, rI1_ca1, rE2_ca1, rI2_ca1, dg_input, ec_input):

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  
  # CA3 plots with DG input
  ax1 = axes[0, 0]
  ax1.plot(t, rE1_ca3, 'b', label='E population', linewidth=2)
  ax1.plot(t, rI1_ca3, 'r', label='I population', linewidth=2)
  ax1.set_ylabel('Activity')
  ax1.set_title('CA3 WC - Initial Condition 1')
  ax1.set_ylim(0, 0.5)  # Consistent y-axis limits
  ax1.legend(loc='best')
  ax1.grid(True, alpha=0.3)

  ax2 = axes[0, 1]
  ax2.plot(t, rE2_ca3, 'b', label='E population', linewidth=2)
  ax2.plot(t, rI2_ca3, 'r', label='I population', linewidth=2)
  ax2.set_title('CA3 WC - Initial Condition 2')
  ax2.set_ylim(0, 0.5)  # Consistent y-axis limits
  ax2.legend(loc='best')
  ax2.grid(True, alpha=0.3)

  # DG input plot
  ax3 = axes[0, 2]
  # Show actual DG input values without normalization
  ax3.plot(t, dg_input, 'g', label='DG Input', linewidth=2)
  ax3.set_xlabel('t (ms)')
  ax3.set_ylabel('DG Input Amplitude')
  ax3.set_title('DG Input to CA3')
  ax3.legend(loc='best')
  ax3.grid(True, alpha=0.3)

  # CA1 plots
  ax5 = axes[1, 0]
  ax5.plot(t, rE1_ca1, 'b', label='E population', linewidth=2)
  ax5.plot(t, rI1_ca1, 'r', label='I population', linewidth=2)
  ax5.set_xlabel('t (ms)')
  ax5.set_ylabel('Activity')
  ax5.set_title('CA1 WC - Initial Condition 1')
  ax5.set_ylim(0, 0.5)  # Consistent y-axis limits
  ax5.legend(loc='best')
  ax5.grid(True, alpha=0.3)

  ax6 = axes[1, 1]
  ax6.plot(t, rE2_ca1, 'b', label='E population', linewidth=2)
  ax6.plot(t, rI2_ca1, 'r', label='I population', linewidth=2)
  ax6.set_xlabel('t (ms)')
  ax6.set_title('CA1 WC - Initial Condition 2')
  ax6.set_ylim(0, 0.5)  # Consistent y-axis limits
  ax6.legend(loc='best')
  ax6.grid(True, alpha=0.3)

  # Combined plot showing DG input and CA3 response
  ax7 = axes[1, 2]
  ax7_twin = ax7.twinx()
  
  # Plot DG input on primary y-axis (actual values)
  line1 = ax7.plot(t, dg_input, 'g', label='DG Input', linewidth=2)
  ax7.set_xlabel('t (ms)')
  ax7.set_ylabel('DG Input Amplitude', color='g')
  ax7.tick_params(axis='y', labelcolor='g')
  
  # Plot CA3 E activity on secondary y-axis
  line2 = ax7_twin.plot(t, rE1_ca3, 'b', label='CA3 E Activity', linewidth=2)
  ax7_twin.set_ylabel('CA3 E Activity', color='b')
  ax7_twin.tick_params(axis='y', labelcolor='b')
  
  ax7.set_title('DG Input vs CA3 Response')
  ax7.grid(True, alpha=0.3)
  
  # Combine legends
  lines = line1 + line2
  labels = [l.get_label() for l in lines]
  ax7.legend(lines, labels, loc='upper right')

  plt.tight_layout()
  plt.show()

def plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI):
  plt.figure()
  plt.plot(Exc_null_rE, Exc_null_rI, 'b', label='E nullcline')
  plt.plot(Inh_null_rE, Inh_null_rI, 'r', label='I nullcline')
  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')
  plt.legend(loc='best')
  # plt.show() removed to allow all plots to be shown at once

def get_E_nullcline(rE, a_E, theta_E, wEE, wEI, ext_E, **other_pars):
  """
  Solve for rI along the rE from drE/dt = 0.

  Args:
    rE    : response of excitatory population
    a_E, theta_E, wEE, wEI, I_ext_E : Wilson-Cowan excitatory parameters
    Other parameters are ignored

  Returns:
    rI    : values of inhibitory population along the nullcline on the rE
  """
  # calculate rI for E nullclines on rI
  rI = 1 / wEI * (wEE * rE - F_inv(rE, a_E, theta_E) + ext_E)

  return rI


def get_I_nullcline(rI, a_I, theta_I, wIE, wII, ext_I, **other_pars):
  """
  Solve for E along the rI from dI/dt = 0.

  Args:
    rI    : response of inhibitory population
    a_I, theta_I, wIE, wII, I_ext_I : Wilson-Cowan inhibitory parameters
    Other parameters are ignored

  Returns:
    rE    : values of the excitatory population along the nullcline on the rI
  """
  # calculate rE for I nullclines on rI
  rE = 1 / wIE * (wII * rI + F_inv(rI, a_I, theta_I) - ext_I)

  return rE


def my_plot_vector(pars, my_n_skip=2, myscale=5):
  EI_grid = np.linspace(0., 1., 20)
  rE, rI = np.meshgrid(EI_grid, EI_grid)
  drEdt, drIdt = EIderivs(rE, rI, **pars)

  n_skip = my_n_skip

  plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
             drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
             angles='xy', scale_units='xy', scale=myscale, facecolor='c')

  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')


def my_plot_trajectory(pars, mycolor, x_init, mylabel):
  pars = pars.copy()
  pars['rE_init'], pars['rI_init'] = x_init[0], x_init[1]
  rE_tj, rI_tj = simulate_wc(**pars)

  plt.plot(rE_tj, rI_tj, color=mycolor, label=mylabel)
  plt.plot(x_init[0], x_init[1], 'o', color=mycolor, ms=8)
  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')


def plot_fp(x_fp, position=(0.02, 0.1), rotation=0):
  plt.plot(x_fp[0], x_fp[1], 'ko', ms=8)
  plt.text(x_fp[0] + position[0], x_fp[1] + position[1],
           f'Fixed Point1=\n({x_fp[0]:.3f}, {x_fp[1]:.3f})',
           horizontalalignment='center', verticalalignment='bottom',
           rotation=rotation)


def EIderivs(rE, rI,
             tau_E, a_E, theta_E, wEE, wEI, ext_E,
             tau_I, a_I, theta_I, wIE, wII, ext_I,
             **other_pars):
  """Time derivatives for E/I variables (dE/dt, dI/dt)."""

  # Compute the derivative of rE
  drEdt = (-rE + F(wEE * rE - wEI * rI + ext_E, a_E, theta_E)) / tau_E

  # Compute the derivative of rI
  drIdt = (-rI + F(wIE * rE - wII * rI + ext_I, a_I, theta_I)) / tau_I

  return drEdt, drIdt

def plot_complete_analysis(pars):
  plt.figure(figsize=(7.7, 6.))

  # plot example trajectories
  my_plot_trajectories(pars, 0.2, 6,
                       'Sample trajectories \nfor different init. conditions')
  my_plot_trajectory(pars, 'orange', [0.6, 0.8],
                     'Sample trajectory for \nlow activity')
  my_plot_trajectory(pars, 'm', [0.6, 0.6],
                     'Sample trajectory for \nhigh activity')

  # plot nullclines
  Exc_null_rE = np.linspace(-0.01, 0.96, 100)
  Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)
  Inh_null_rI = np.linspace(-0.01, 0.96, 100)
  Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)
  plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI)

  # plot fixed points
  fp_1 = my_fp(pars, 0.5, 0.2)
  if check_fp(pars, fp_1):
      plot_fp(fp_1)

  # check stability of fixed points
  eig_1 = get_eig_Jacobian(fp_1, **pars)
  print("Eigenvalues of fixed point 1:", eig_1)

  # plot vector field
  EI_grid = np.linspace(0., 1., 20)
  rE, rI = np.meshgrid(EI_grid, EI_grid)
  drEdt, drIdt = EIderivs(rE, rI, **pars)
  n_skip = 2
  plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
             drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
             angles='xy', scale_units='xy', scale=5., facecolor='c')

  plt.legend(loc=[1.02, 0.57], handlelength=1)
  plt.show()

def my_plot_trajectories(pars, dx, n, mylabel):
  """
  Solve for I along the E_grid from dE/dt = 0.

  Expects:
  pars    : Parameter dictionary
  dx      : increment of initial values
  n       : n*n trjectories
  mylabel : label for legend

  Returns:
    figure of trajectory
  """
  pars = pars.copy()
  for ie in range(n):
    for ii in range(n):
      pars['rE_init'], pars['rI_init'] = dx * ie, dx * ii
      rE_tj, rI_tj = simulate_wc(**pars)
      if (ie == n-1) & (ii == n-1):
          plt.plot(rE_tj, rI_tj, 'gray', alpha=0.8, label=mylabel)
      else:
          plt.plot(rE_tj, rI_tj, 'gray', alpha=0.8)

  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')

def simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I,
                wEE, wEI, wIE, wII, ext_E, ext_I,
                rE_init, rI_init, dt, range_t, **other_pars):
        """
        Run a single simulation with the given parameters
        """
        # initialize activity arrays
        Lt = range_t.size

        # initialize initial rates then fill in 0s for the rest 
        rE = np.append(rE_init, np.zeros(Lt - 1))
        rI = np.append(rI_init, np.zeros(Lt - 1))
        
        # initialize external inputs
        ext_E = ext_E * np.ones(Lt)
        ext_I = ext_I * np.ones(Lt)

        # simulate the Wilson-Cowan equations
        for k in range(Lt - 1):

            # Calculate the derivative of the E population
            drE = dt / tau_E * (-rE[k] + F(wEE * rE[k] - wEI * rI[k] + ext_E[k], a_E, theta_E))

            # Calculate the derivative of the I population
            drI = dt / tau_I * (-rI[k] + F(wIE * rE[k] - wII * rI[k] + ext_I[k],
                                        a_I, theta_I))

            # Update using Euler's method
            rE[k + 1] = rE[k] + drE
            rI[k + 1] = rI[k] + drI

        # return arrays with all the rates over time
        return rE, rI

def my_fp(pars, rE_init, rI_init):
  """
  Use opt.root function to solve Equations (2)-(3) from initial values
  """

  tau_E, a_E, theta_E = pars['tau_E'], pars['a_E'], pars['theta_E']
  tau_I, a_I, theta_I = pars['tau_I'], pars['a_I'], pars['theta_I']
  wEE, wEI = pars['wEE'], pars['wEI']
  wIE, wII = pars['wIE'], pars['wII']
  ext_E, ext_I = pars['ext_E'], pars['ext_I']

  # define the right hand of wilson-cowan equations
  def my_WCr(x):

    rE, rI = x
    drEdt = (-rE + F(wEE * rE - wEI * rI + ext_E, a_E, theta_E)) / tau_E
    drIdt = (-rI + F(wIE * rE - wII * rI + ext_I, a_I, theta_I)) / tau_I
    y = np.array([drEdt, drIdt])

    return y

  x0 = np.array([rE_init, rI_init])
  x_fp = opt.root(my_WCr, x0).x

  return x_fp

def find_multiple_fixed_points(pars, initial_conditions=None):
  """
  Find multiple fixed points using different initial conditions.
  
  Args:
    pars: parameter dictionary
    initial_conditions: list of [rE_init, rI_init] pairs. If None, uses default grid.
  
  Returns:
    list of unique fixed points
  """
  if initial_conditions is None:
    # Create a grid of initial conditions
    rE_vals = np.linspace(0.1, 0.9, 5)
    rI_vals = np.linspace(0.1, 0.9, 5)
    initial_conditions = []
    for rE in rE_vals:
      for rI in rI_vals:
        initial_conditions.append([rE, rI])
  
  fixed_points = []
  
  for rE_init, rI_init in initial_conditions:
    try:
      fp = my_fp(pars, rE_init, rI_init)
      if check_fp(pars, fp):
        # Check if this fixed point is already found (within tolerance)
        is_duplicate = False
        for existing_fp in fixed_points:
          if np.linalg.norm(fp - existing_fp) < 1e-3:
            is_duplicate = True
            break
        
        if not is_duplicate:
          fixed_points.append(fp)
    except:
      # Skip if optimization fails
      continue
  
  return fixed_points

def check_fp(pars, x_fp, mytol=1e-6):
  """
  Verify (drE/dt)^2 + (drI/dt)^2< mytol

  Args:
    pars    : Parameter dictionary
    fp      : value of fixed point
    mytol   : tolerance, default as 10^{-6}

  Returns :
    Whether it is a correct fixed point: True/False
  """

  drEdt, drIdt = EIderivs(x_fp[0], x_fp[1], **pars)

  return drEdt**2 + drIdt**2 < mytol

def get_eig_Jacobian(fp,
                     tau_E, a_E, theta_E, wEE, wEI, ext_E,
                     tau_I, a_I, theta_I, wIE, wII, ext_I, **other_pars):
  """Compute eigenvalues of the Wilson-Cowan Jacobian matrix at fixed point."""
  # Initialization
  rE, rI = fp
  J = np.zeros((2, 2))

  # Compute the four elements of the Jacobian matrix
  J[0, 0] = (-1 + wEE * dF(wEE * rE - wEI * rI + ext_E,
                           a_E, theta_E)) / tau_E

  J[0, 1] = (-wEI * dF(wEE * rE - wEI * rI + ext_E,
                       a_E, theta_E)) / tau_E

  J[1, 0] = (wIE * dF(wIE * rE - wII * rI + ext_I,
                      a_I, theta_I)) / tau_I

  J[1, 1] = (-1 - wII * dF(wIE * rE - wII * rI + ext_I,
                           a_I, theta_I)) / tau_I

  # Compute and return the eigenvalues
  evals = np.linalg.eig(J)[0]
  return evals

def plot_bifurcation_diagram_multiple_fps(parameter_values, all_fixed_points, all_stabilities, parameter_name='wEE'):
  """
  Plot bifurcation diagram with multiple fixed points per parameter value.
  
  Args:
    parameter_values: array of parameter values (x-axis)
    all_fixed_points: list of lists, each inner list contains fixed points for one parameter value
    all_stabilities: list of lists, each inner list contains stabilities for one parameter value
    parameter_name: name of the parameter being varied
  """
  plt.figure(figsize=(12, 8))
  
  # Separate stable and unstable fixed points
  stable_params = []
  stable_fps_E = []
  stable_fps_I = []
  unstable_params = []
  unstable_fps_E = []
  unstable_fps_I = []
  
  for param_val, fps, stabs in zip(parameter_values, all_fixed_points, all_stabilities):
    for fp, is_stable in zip(fps, stabs):
      if is_stable:
        stable_params.append(param_val)
        stable_fps_E.append(fp[0])  # E component
        stable_fps_I.append(fp[1])  # I component
      else:
        unstable_params.append(param_val)
        unstable_fps_E.append(fp[0])  # E component
        unstable_fps_I.append(fp[1])  # I component
  
  # Plot E component (circles)
  if stable_params:
    plt.plot(stable_params, stable_fps_E, 'go', markersize=10, label='E* (stable)', alpha=0.8)
  if unstable_params:
    plt.plot(unstable_params, unstable_fps_E, 'ro', markersize=10, label='E* (unstable)', alpha=0.8)
  
  # Plot I component (triangles) with lighter colors
  if stable_params:
    plt.plot(stable_params, stable_fps_I, '^', color='lightgreen', markersize=10, label='I* (stable)', alpha=0.8)
  if unstable_params:
    plt.plot(unstable_params, unstable_fps_I, '^', color='lightcoral', markersize=10, label='I* (unstable)', alpha=0.8)
  
  plt.xlabel(parameter_name)
  plt.ylabel('Fixed Point Value')
  plt.title(f'Bifurcation Diagram: {parameter_name}')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()

