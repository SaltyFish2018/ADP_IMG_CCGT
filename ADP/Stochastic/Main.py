import numpy as np
from Load_Data import Load_Data
from MC import Guass_Generation, Weibull_Generation
from Scenario import Environment


if __name__ == '__main__' :
    # deterministic data
    types_par, demand_Elec_original, demand_Heat_original, P_WT_a_original, price_original = Load_Data()

    for n in range(1,3):
        # stochasitic scences (mean, variance/ shape, scale)
        demand_Elec = Guass_Generation(demand_Elec_original, 0, 0.1)
        demand_Heat = Guass_Generation(demand_Heat_original, 0, 0.1)
        price = Guass_Generation(price_original, 0, 0.00001)
        P_WT_a = Weibull_Generation(P_WT_a_original, 1, 0.1)





