# Neural Mass Model of the Hippocampus

## Details
This repository computationally models the regions of the hippocampus.
<br>
<br>
Specifically, it uses [Wilson Cowan Equations](https://pubmed.ncbi.nlm.nih.gov/4332108/) to model properties of populations of neurons in the Dentate Gyrus (DG), the Cornu Ammonis 3 (CA3), and the Cornu Ammonis 1 (CA1). Model parameters were loosely set based on biophysical properties of each region and are adaptable. 

### Wilson-Cowan Update Equations
To model the change in neuronal firing rates for the excitatory population:

$$\Delta r_E = \frac{dt}{\tau_E} \left( -r_E[k] + F(w_{EE} \cdot r_E[k] - w_{EI} \cdot r_I[k] + \text{ext}_E[k], a_E, \theta_E) \right)$$

To model the change in neuronal firing rates for the inhibitory population:

$$\Delta r_I = \frac{dt}{\tau_I} \left( -r_I[k] + F(w_{IE} \cdot r_E[k] - w_{II} \cdot r_I[k] + \text{ext}_I[k], a_I, \theta_I) \right)$$

To model acetylcholine, we modulate wEE (recurrent excitation) and a_E (gain) as follows:
$$\Delta r_E = \frac{dt}{\tau_E} \left( -r_E[k] + F(w_{EE} \cdot (1 - ACh) \cdot r_E[k] - w_{EI} \cdot r_I[k] + \text{ext}_E[k], a_E \cdot (1 + ACh), \theta_E) \right)$$

## Scripts

### `python3 trisynaptic.py --theta_osc --ach --ec_E --ec_I`
This script simulates the entire trisynaptic circuit of the hippocampus. It consists of 3 feedforward Wilson Cowan Models that simulate the DG, CA3 and CA1 regions of the hippocampus, driven by a simulated external theta modulated input from the entorhinal cortex (EC). The flow of the trisynaptic circuit is: EC -> DG -> CA3 -> CA1.

`--theta_osc` (boolean) [optional] Add theta oscillations to simulate the input from the entorhinal cortex (EC) to the dentate gyrus (DG).
`--ach` (boolean) [optional] Inject acetylcholine into all three regions (DG, CA3, CA1) to model diffuse effects.
`--ec_E` [optional] Control the magnitude of the external input (Gaussian noise) from EC to the excitatory population if not configuring for theta oscillations.
`--ec_I` [optional] Control the magnitude of the external input (Gaussian noise) from EC to the inhibitory population if not configuring for theta oscillations.

### `python3 simulate.py --region [dg, ca3, ca1] --is_DG_input --is_acetylcholine --ext_E --ext_I --ic_rE --ic_rI`
This script simulates a specified region of the hippocampus (i.e. DG, CA3, CA1). It solves the Wilson Cowan differential equations using Euler's method over discrete time steps. It generates several result plots: mean firing rates over time, mean firing rates across different parameter values, nullclines, bifurcations, and prints out the system's fixed points.

`--region` Specify the region to simulate (dg, ca3, ca1).
`--ach` (boolean) [optional] Add acetylcholine to the region. It models a sigmoidal change in ACh as a function of time.
`--dg_input` (boolean) [optional] Simulate an external input from the dentate gyrus (DG) (used for CA3 simulations), which is modeled as infrequent high amplitude pulses. Setting this flag also by default simulates an external input from the entorhinal cortex (EC) to CA3 which is modeled as a continuous low-amplitude noisy input.


`--ext_E` [optional] Specify the external input to the excitatory population (default = 0.5).
`--ext_I` [optional] Specifiy the external input to the inhibitory population (default = 0.5).
`--ic_rE` [optional] Specify the initial average firing rate of the excitatory population (default = 0.32).
`--ic_rI` [optional] Specify the initial average firing rate of the inhibitory population (default = 0.15).
`--ic_rI` [optional] Specify a param to vary (can edit range within code) (default = wEE).

### `python3 2d_bifurcation.py --region [dg, ca3, ca1] --param1 --param2 --min1 --max1 --min2 --max2 --ext_E --ext_I` This script plots a 2D Bifurcation analysis of the two given parameters. You can specify a range for each parameter (i.e. `--min1` to `--max1`) to plot the combinations. It classifies fixed points into stable/unstable, oscillatory, saddle. 

## Requirements
`pip install -r requirements`

## Sample Plots
### Trisynaptic Circuit Simulation
![Trisynaptic Circuit](/sample_plots/trisynaptic_plot.png)
`python3 trisynaptic.py --theta_osc --ach`

### Simulating CA3
![CA3 Firing Rates](/sample_plots/ca3_frs.png)
`python3 simulate.py --region ca3`

### Trisynaptic Circuit Simulation
![2D Bifurcation](/sample_plots/bifurcation.png)
`python3 2d_bifurcation.py --param1 wEE --param2 a_E --min1 0 --max1 15 --min2 0 --max2 15  --region ca3`


## References
- [Oscillatory dynamics in the hippocampus support dentate gyrus–CA3 coupling (Akam et al.)](https://www.nature.com/articles/nn.3081)
