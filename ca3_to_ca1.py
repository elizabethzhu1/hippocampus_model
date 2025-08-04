import numpy as np
import matplotlib.pyplot as plt
from helpers import set_ca3_parameters, set_ca1_parameters, my_test_plot_ca3_to_ca1_with_dg
from models.ca3 import WilsonCowan
from models.ca1 import CA1_WilsonCowan

def main():

    ca3_pars = set_ca3_parameters()
    ca3_model = WilsonCowan(**ca3_pars)

    # Get CA3 activity and DG/EC inputs
    rE1_ca3, rI1_ca3, dg_input1, ec_input1 = ca3_model.simulate(rE_init=0.33, rI_init=0.15)
    rE2_ca3, rI2_ca3, dg_input2, ec_input2 = ca3_model.simulate(rE_init=0.0, rI_init=0.0)

    ca1_pars = set_ca1_parameters()  # Use CA1-specific parameters
    ca1_pars['schaffer_input'] = rE1_ca3  # Get the excitatory firing rate from CA3 from one initial condition
    
    ca1_model = CA1_WilsonCowan(**ca1_pars)

    rE1_ca1, rI1_ca1 = ca1_model.simulate(rE_init=0.33, rI_init=0.15)
    rE2_ca1, rI2_ca1 = ca1_model.simulate(rE_init=0.0, rI_init=0.0)

    # Use the new plotting function that includes DG and EC inputs
    my_test_plot_ca3_to_ca1_with_dg(ca3_pars['range_t'], rE1_ca3, rI1_ca3, rE2_ca3, rI2_ca3, 
                                     rE1_ca1, rI1_ca1, rE2_ca1, rI2_ca1, dg_input1, ec_input1)



if __name__ == "__main__":
    main()
