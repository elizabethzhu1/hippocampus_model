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
`python3 simulate.py --is_DG_input --is_acetylcholine --is_adaptation` 
<br>
`python3 ca3_to_ca1.py --dg_input_ca3 --ach_ca3 --adaptation_ca3 --ach_ca1 --adaptation_ca1`
<br>

`python3 simulate.py --is_DG_input --is_acetylcholine --is_adaptation`
<br>
This script solves the Wilson Cowan differential equations using Euler's method over discrete time steps. Furthermore, it will plot many visualizations, such as the change in firing rate results over many values of the same parameter, a bifurcation diagram illustrating the senstivity of the parameter, nullclines and fixed points, and it allows you to compare standard vs. customized WC models. You can specify whether to include an input from DG, acetylcholine (varied over time), and an adaptation current.
<br>

`python3 ca3_to_ca1.py --dg_input_ca3 --ach_ca3 --adaptation_ca3 --ach_ca1 --adaptation_ca1`
<br>
This script visualizes the effects of CA3 on the CA1 region of the hippocampus. 
