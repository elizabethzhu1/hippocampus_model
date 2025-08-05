# Wilson Cowan Models of Acetylcholine in the Hippocampus (CA3 / CA1)

## Details
This repository is a basic model of the CA3 and CA1 regions of the hippocampus using Wilson Cowan Equations.
<br>
<br>
Parameters for the Wilson Cowan Equations were modified based on biophysical properites of CA3 and CA1 respectively. For CA3, I used the following [paper](https://www.nature.com/articles/nn.3081) by Akam et al. The parameters for CA1 were loosely based off its biophysical similarities / differences relative to CA3.

### Wilson-Cowan Update Equations
To model the change in neuronal firing rates for the excitatory population:

$$\Delta r_E = \frac{dt}{\tau_E} \left( -r_E[k] + F(w_{EE} \cdot r_E[k] - w_{EI} \cdot r_I[k] + \text{ext}_E[k], a_E, \theta_E) \right)$$

To model the change in neuronal firing rates for the inhibitory population:

$$\Delta r_I = \frac{dt}{\tau_I} \left( -r_I[k] + F(w_{IE} \cdot r_E[k] - w_{II} \cdot r_I[k] + \text{ext}_I[k], a_I, \theta_I) \right)$$

## Scripts
<!-- `python3 simulate.py --is_DG_input --is_acetylcholine --is_adaptation` 
<br>
`python3 ca3_to_ca1.py --dg_input_ca3 --ach_ca3 --adaptation_ca3 --ach_ca1 --adaptation_ca1`
<br> -->

1. `python3 simulate.py --is_DG_input --is_acetylcholine --is_adaptation`
<br>
This script solves the Wilson Cowan differential equations using Euler's method over discrete time steps. It plots visualizations of the change in firing rate results over changes in the same parameter, a bifurcation diagram illustrating the sensitivity of the parameter, nullclines and fixed points, and it allows you to compare standard vs. customized WC models. You can include various features such as external inputs from DG, acetylcholine (varied over time), and an adaptation current. 
<br
`--is_DG_input'> This flag simulates an external input from the dentate gyrus (DG) to CA3, which is modeled as infrequent high amplitude pulses. Setting this flag also by default simulates an external input from the entorhinal cortex (EC) to CA3 which is modeled as a continuous low-amplitude noisy input
<br
`--is_acetylcholine`> This flag models a change in acetylcholine over time, in the shape of a sigmoid. Notably, increasing acetylcholine linearly decreases the wEE parameter and increases the a_E parameter. 
<br `--is_adaptation`> This flag adds an adaptation current to the Wilson Cowan equations. 
<br>
<br>

2. `python3 ca3_to_ca1.py --dg_input_ca3 --ach_ca3 --adaptation_ca3 --ach_ca1 --adaptation_ca1`
This script visualizes the effects of CA3 on the CA1 region of the hippocampus. 
