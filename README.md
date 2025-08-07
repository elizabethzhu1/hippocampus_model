# Wilson Cowan Models of Acetylcholine in the Hippocampus (CA3 / CA1)

## Details
This repository computationally models the regions of the hippocampus.
<br>
<br>
Specifically, it uses Wilson Cowan Equations to model the Dentate Gyrus (DG), the Cornu Ammonis 3 (CA3), and the Cornu Ammonis 1 (CA1).
<br>
<br>
Parameters fare set based on biophysical properites of each region (see the references section below). For CA3, I used the following [paper](https://www.nature.com/articles/nn.3081) by Akam et al. The parameters for CA1 were loosely based off its biophysical similarities / differences relative to CA3.

### Wilson-Cowan Update Equations
To model the change in neuronal firing rates for the excitatory population:

$$\Delta r_E = \frac{dt}{\tau_E} \left( -r_E[k] + F(w_{EE} \cdot r_E[k] - w_{EI} \cdot r_I[k] + \text{ext}_E[k], a_E, \theta_E) \right)$$

To model the change in neuronal firing rates for the inhibitory population:

$$\Delta r_I = \frac{dt}{\tau_I} \left( -r_I[k] + F(w_{IE} \cdot r_E[k] - w_{II} \cdot r_I[k] + \text{ext}_I[k], a_I, \theta_I) \right)$$

To model acetylcholine, we modulate wEE (recurrent excitation) and a_E (gain) as follows:
$$\Delta r_E = \frac{dt}{\tau_E} \left( -r_E[k] + F(w_{EE} \cdot (1 - ACh) \cdot r_E[k] - w_{EI} \cdot r_I[k] + \text{ext}_E[k], a_E \cdot (1 + ACh), \theta_E) \right)$$

## Scripts

1. `python3 trisynaptic.py --theta_osc`
This script simulates the entire trisynaptic circuit of the hippocampus.

2. `python3 simulate.py --region [dg, ca3, ca1] --is_DG_input --is_acetylcholine --is_adaptation`
This script simulates a specified region of the hippocampus. It solves the Wilson Cowan differential equations using Euler's method over discrete time steps. 

It plots visualizations of the change in firing rate results over changes in the same parameter, a bifurcation diagram illustrating the sensitivity of the parameter, nullclines and fixed points, and it allows you to compare standard vs. customized WC models. You can include various features such as external inputs from DG, acetylcholine (varied over time), and an adaptation current. 

`--is_DG_input` This boolean flag simulates an external input from the dentate gyrus (DG) to CA3, which is modeled as infrequent high amplitude pulses. Setting this flag also by default simulates an external input from the entorhinal cortex (EC) to CA3 which is modeled as a continuous low-amplitude noisy input

`--is_acetylcholine` This boolean flag adds acetylcholine to the region. It models a sigmoidal change in ACh as a function of time.

`--is_adaptation` This boolean flag adds an adaptation current to the Wilson Cowan equations. 

### Optional
`--ext_E` Specify the external input to the excitatory population (default = 0.5).
`--ext_I` Specifiy the external input to the inhibitory population (default = 0.5).
`--ic_E` Specify the initial average firing rate of the excitatory population.
`--ic_I` Specify the initial average firing rate of the inhibitory population.

<br>
<br>

3. `python3 ca3_to_ca1.py --dg_input_ca3 --ach_ca3 --adaptation_ca3 --ach_ca1 --adaptation_ca1`
This script simulates the effects of CA3 on the CA1 region of the hippocampus. CA3 and CA1 are each represented by a Wilson-Cowan Model, each with their own custom paramters. Setting the acetylcholine flag simulates a release in both regions.


## References

DG Parameters
- 

CA3 Parameters
- [Akam et al.](https://www.nature.com/articles/nn.3081)

CA1 Parameters
-