import numpy as np
import matplotlib.pyplot as plt
from helpers import set_ca3_parameters, set_ca1_parameters, my_test_plot_ca3_to_ca1_with_dg
from models.ca3 import WilsonCowan
from models.ca1 import CA1_WilsonCowan
import argparse


def main(args):

    ca3_pars = set_ca3_parameters()

    # Set boolean flags for features
    if args.dg_input_ca3:
        ca3_pars['is_DG_input'] = True
    if args.ach_ca3:
        ca3_pars['is_acetylcholine'] = True
    if args.adaptation_ca3:
        ca3_pars['is_adaptation'] = True

    ca3_model = WilsonCowan(**ca3_pars)

    # Get CA3 activity and DG/EC inputs
    rE1_ca3, rI1_ca3, dg_input1, ec_input1 = ca3_model.simulate(rE_init=0.33, rI_init=0.15)
    rE2_ca3, rI2_ca3, dg_input2, ec_input2 = ca3_model.simulate(rE_init=0.0, rI_init=0.0)

    ca1_pars = set_ca1_parameters()  # Use CA1-specific parameters

    if args.ach_ca1:
        ca1_pars['is_acetylcholine'] = True
    if args.adaptation_ca1:
        ca1_pars['is_adaptation'] = True
    ca1_pars['schaffer_input'] = rE1_ca3  # Get the excitatory firing rate from CA3 from one initial condition

    ca1_model = CA1_WilsonCowan(**ca1_pars)

    rE1_ca1, rI1_ca1 = ca1_model.simulate(rE_init=0.33, rI_init=0.15)
    rE2_ca1, rI2_ca1 = ca1_model.simulate(rE_init=0.0, rI_init=0.0)

    # Use the new plotting function that includes DG and EC inputs
    my_test_plot_ca3_to_ca1_with_dg(ca3_pars['range_t'], rE1_ca3, rI1_ca3, rE2_ca3, rI2_ca3, 
                                     rE1_ca1, rI1_ca1, rE2_ca1, rI2_ca1, dg_input1, ec_input1)


if __name__ == "__main__":

    # Introduce arguments
    parser = argparse.ArgumentParser(description='Simulate the Wilson-Cowan model')
    parser.add_argument('--dg_input_ca3', action='store_true', help='Include DG input')
    parser.add_argument('--ach_ca3', action='store_true', help='Include ACh modulation')
    parser.add_argument('--adaptation_ca3', action='store_true', help='Include adaptation')

    parser.add_argument('--ach_ca1', action='store_true', help='Include ACh modulation')
    parser.add_argument('--adaptation_ca1', action='store_true', help='Include adaptation')

    args = parser.parse_args()

    main(args)
