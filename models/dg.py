import numpy as np
from helpers import F
import matplotlib.pyplot as plt

"""
τ_E * drE/dt = -rE + F(wEE*rE - wEI*rI + ext_E)
τ_I * drI/dt = -rI + F(wIE*rE - wII*rI + ext_I)

rE, rI (arrays): firing rates of excitatory and inhibitory populations
wEE, wEI, wIE, wII (floats): weights of the connections
tau_E, tau_I (floats): time constants of the excitatory and inhibitory populations
a_E, a_I (floats): gain parameters of the excitatory and inhibitory populations (excitability)
theta_E, theta_I (floats): threshold parameters of the excitatory and inhibitory populations
ext_E, ext_I (arrays): external inputs to the excitatory and inhibitory populations
dt (float): time step
range_t (array): time points

Solving Method -- Euler's Method
    rE[k + 1] = rE[k] + drE
    rI[k + 1] = rI[k] + drI

Run simulations with different initial conditions (rE_init, rI_init)
"""

class DG_WilsonCowan:
    def __init__(self, wEE, wEI, wIE, wII, tau_E, tau_I, a_E, a_I, theta_E, theta_I, rE_init, rI_init, dt, range_t, T, tau_A, ext_E, ext_I, is_adaptation=False, is_acetylcholine=False):
        self.wEE = wEE 
        self.wEI = wEI
        self.wIE = wIE
        self.wII = wII
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.a_E = a_E
        self.a_I = a_I
        self.theta_E = theta_E
        self.theta_I = theta_I
        self.rE_init = rE_init
        self.rI_init = rI_init
        self.dt = dt
        self.range_t = range_t 
        self.T = T
        self.tau_A = tau_A
        self.ext_E = ext_E
        self.ext_I = ext_I

        # boolean flags for features
        self.is_adaptation = is_adaptation
        self.is_acetylcholine = is_acetylcholine


    def simulate(self, **kwargs):
        """
        Run a single simulation with the given parameters
        """
        # Update instance parameters with any provided kwargs
        if 'rE_init' in kwargs:
            rE_init = kwargs['rE_init']
        else:
            rE_init = self.rE_init
            
        if 'rI_init' in kwargs:
            rI_init = kwargs['rI_init']
        else:
            rI_init = self.rI_init

        # initialize activity arrays
        Lt = self.range_t.size

        # initialize initial rates then fill in 0s for the rest 
        rE = np.append(rE_init, np.zeros(Lt - 1))
        rI = np.append(rI_init, np.zeros(Lt - 1))

        tau_E = self.tau_E
        tau_I = self.tau_I
        theta_E = self.theta_E
        theta_I = self.theta_I
        wEE = self.wEE
        wEI = self.wEI
        wIE = self.wIE
        wII = self.wII
        a_E = self.a_E
        a_I = self.a_I
        dt = self.dt
        tau_A = self.tau_A
        ext_E = self.ext_E
        ext_I = self.ext_I

        # Initialize adaptation starting rates
        A = np.append(0, np.zeros(Lt - 1))

        # Simulate the Wilson-Cowan equations
        for k in range(Lt - 1):

            # Calculate the derivative of the E population
            drE = dt / tau_E * (-rE[k] + F(wEE * rE[k] - wEI * rI[k] + ext_E[k], a_E, theta_E))

            # Calculate the derivative of the I population
            drI = dt / tau_I * (-rI[k] + F(wIE * rE[k] - wII * rI[k] + ext_I[k], a_I, theta_I))

            if self.is_adaptation and self.is_acetylcholine:
                drE = dt / tau_E * (-rE[k] + F(self.ACh_modulation_wEE(wEE, k) * rE[k] - wEI * rI[k] + ext_E[k] + A[k], a_E, theta_E))

                # calculate adaptation variable
                drA = - dt / tau_A * (rE[k] - A[k])
            
            # Modify equation based on features -- can stack multiple
            if self.is_adaptation:
                drE = dt / tau_E * (-rE[k] + F(wEE * rE[k] - wEI * rI[k] + ext_E[k] + A[k], a_E, theta_E))

                # calculate adaptation variable
                drA = - dt / tau_A * (rE[k] - A[k])

            if self.is_acetylcholine:
                # modulate wEE (decrease) + a_E (increase)
                drE = dt / tau_E * (-rE[k] + F(self.ACh_modulation_wEE(wEE, k) * rE[k] - wEI * rI[k] + ext_E[k], a_E, theta_E))

            # Add noise into the system
            noise_E = np.random.normal(0, 0.001)
            noise_I = np.random.normal(0, 0.001)

            # Update using Euler's method
            rE[k + 1] = rE[k] + drE + noise_E
            rI[k + 1] = rI[k] + drI + noise_I

            # add adaptation variable
            if self.is_adaptation:      
                A[k + 1] = A[k] + drA

        # return arrays with all the rates over time
        return rE, rI

    
    def ACh_modulation_wEE(self, wEE, t):
        # modulate wEE with ACh
        ach_value = self.ACh_func(t)
        modulated_wEE = wEE * (1 - ach_value)
        return modulated_wEE


    def ACh_modulation_a_E(self, a_E, t):
        # modulate a_E with ACh
        ach_value = self.ACh_func(t)
        modulated_a_E = a_E * (1 + ach_value)
        return modulated_a_E
    
    
    def ACh_modulation_ext_E(self, ext_E, t):
        # modulate external input with ACh
        ach_value = self.ACh_func(t)
        modulated_ext_E = ext_E[t] * (1 + ach_value)
        return modulated_ext_E


    def ACh_func(self, t):
        # ACh function that starts low, ramps up quickly and stays high (sigmoid)
        t0 = 6000  # Time at which ramp begins
        k = 0.001  # Steepness of the ramp
        return 0 + 0.8 / (1 + np.exp(-k * (t - t0)))  # Sigmoid from 0.1 to 0.9


    def plot_activity_with_ach(self, rE, rI, title="CA3 Activity with ACh Modulation"):
        """
        Plot excitatory and inhibitory activity with ACh modulation overlaid
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot neural activity
        ax1.plot(self.range_t, rE, 'b', label='Excitatory (E)', linewidth=2)
        ax1.plot(self.range_t, rI, 'r', label='Inhibitory (I)', linewidth=2)
        ax1.set_ylabel('Firing Rate')
        ax1.set_title(title)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Calculate and plot ACh modulation over time
        ach_values = [self.ACh_func(t) for t in range(len(self.range_t))]
        ach_modulation = [self.wEE * (1 - ach) for ach in ach_values]
        
        # Plot ACh function and modulation
        ax2.plot(self.range_t, ach_values, 'g', label='ACh Level', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.range_t, ach_modulation, 'purple', label='Modulated wEE', linewidth=2)
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('ACh Level', color='g')
        ax2_twin.set_ylabel('Modulated wEE', color='purple')
        ax2.legend(loc='upper right')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def theta_modulation(self, input, theta_freq):
        # Modulate external input with theta oscillations
        modulated_input = input * np.sin(2 * np.pi * theta_freq * self.range_t)

        return modulated_input

    
    def EC_input(self):
        # EC input is a low amplitude, continuous input
        baseline = 0.5

        num_ms = int(self.T)
        
        # Generate noise for each millisecond
        ms_noise = np.random.normal(0, 0.01, num_ms)
        ec_input_ms = baseline + ms_noise
        
        # Interpolate to match the time step resolution
        ms_times = np.arange(0, num_ms)
        step_times = self.range_t
        
        # Use linear interpolation to get values at each time step
        ec_input = np.interp(step_times, ms_times, ec_input_ms)

        return ec_input
        