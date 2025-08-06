# Hippocampal Trisynaptic Circuit Model

This repository contains a comprehensive computational model of the hippocampal trisynaptic circuit: **EC → DG → CA3 → CA1**.

## Overview

The trisynaptic circuit is the main excitatory pathway through the hippocampus, consisting of:

1. **Dentate Gyrus (DG)**: Receives input from entorhinal cortex (EC) via the perforant path
2. **CA3**: Receives strong, sparse input from DG via mossy fibers, has strong recurrent connections
3. **CA1**: Receives input from CA3 via Schaffer collaterals and direct input from EC

## Key Features

### Anatomical Accuracy
- **DG**: Sparse firing patterns, strong mossy fiber outputs to CA3
- **CA3**: Strong recurrent connections, receives DG input, ACh modulation
- **CA1**: Receives both CA3 input (Schaffer collaterals) and direct EC input

### Physiological Features
- **ACh Modulation**: Different timing for CA3 vs CA1
- **Sparse Coding**: DG shows sparse, burst-like activity
- **Recurrent Dynamics**: CA3 exhibits rich recurrent dynamics
- **Input Integration**: CA1 combines multiple input streams

## Files Structure

```
hippocampus_model/
├── trisynaptic_circuit.py      # Main circuit model
├── trisynaptic_example.py      # Demonstration scripts
├── models/
│   ├── ca3.py                  # CA3 Wilson-Cowan model
│   ├── ca1.py                  # CA1 Wilson-Cowan model
│   └── dg.py                   # DG Wilson-Cowan model
├── ca3_ach.py                  # CA3 ACh analysis
├── ca3_to_ca1.py              # CA3-CA1 connection
└── helpers.py                  # Parameter setting functions
```

## Usage

### Basic Usage

```python
from trisynaptic_circuit import TrisynapticCircuit

# Create circuit with all features
circuit = TrisynapticCircuit(
    enable_ach=True,
    enable_dg_input=True,
    enable_adaptation=False
)

# Simulate
results = circuit.simulate()

# Plot results
fig1 = circuit.plot_circuit_activity(results)
fig2 = circuit.plot_phase_space(results)

# Analyze dynamics
analysis = circuit.analyze_circuit_dynamics(results)
```

### Running Examples

```bash
# Run the main trisynaptic circuit
python trisynaptic_circuit.py

# Run comprehensive demonstrations
python trisynaptic_example.py

# Run CA3 ACh analysis
python ca3_ach.py
```

## Model Components

### DG (Dentate Gyrus)
- **Function**: Pattern separation, sparse coding
- **Input**: EC via perforant path
- **Output**: Strong mossy fiber projections to CA3
- **Features**: Sparse firing, low baseline activity

### CA3
- **Function**: Pattern completion, autoassociative memory
- **Input**: DG mossy fibers + EC perforant path
- **Output**: Schaffer collaterals to CA1
- **Features**: Strong recurrent connections, ACh modulation

### CA1
- **Function**: Output integration, context integration
- **Input**: CA3 Schaffer collaterals + direct EC input
- **Output**: To subiculum and back to EC
- **Features**: Integrates multiple input streams, ACh modulation

## ACh Modulation

Acetylcholine modulates the circuit at different time scales:

- **CA3 ACh**: Ramps up around 6000ms, affects recurrent connections
- **CA1 ACh**: Ramps up around 18000ms, affects excitability
- **Effect**: Reduces recurrent excitation (wEE modulation)

## Key Parameters

### DG Parameters
```python
tau_E = 2.0      # Slower dynamics
a_E = 0.8        # Lower excitability
theta_E = 3.2    # Higher threshold
wEE = 3.0        # Weak recurrent excitation
```

### CA3 Parameters
```python
tau_E = 1.8      # Moderate dynamics
wEE = 10         # Strong recurrent excitation
wEI = 10         # Strong inhibition
```

### CA1 Parameters
```python
tau_E = 1.6      # Faster dynamics
# Receives CA3 input via schaffer_input parameter
```

## Analysis Features

### Circuit Activity
- Time series plots for all regions
- Input-output relationships
- ACh modulation effects

### Phase Space Analysis
- Trajectories in E-I space for each region
- Start and end points marked

### Statistical Analysis
- Mean and peak firing rates
- Cross-correlations between regions
- Input-output relationships

## Biological Relevance

### Pattern Separation (DG)
- Sparse coding reduces overlap between similar inputs
- Mossy fiber synapses are among the strongest in the brain

### Pattern Completion (CA3)
- Recurrent connections allow pattern completion
- ACh reduces recurrent excitation during encoding

### Integration (CA1)
- Combines CA3 output with direct EC input
- Provides context-dependent output

## Extensions

The model can be extended with:

1. **Theta Oscillations**: Add theta modulation to inputs
2. **Spike Timing**: Convert to spiking neuron models
3. **Plasticity**: Add synaptic plasticity mechanisms
4. **Multiple Subregions**: Model subregions within each area
5. **Learning**: Add learning rules for memory formation

## Dependencies

```python
numpy
matplotlib
scipy  # For correlation analysis
```

## References

- Wilson, H.R. & Cowan, J.D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons.
- Henze, D.A. et al. (2000). Intracellular recordings suggest that mossy fiber synapses are among the strongest in the brain.
- Hasselmo, M.E. (2006). The role of acetylcholine in learning and memory.

## License

This code is provided for educational and research purposes. 