import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from MC import Guass_Generation, Weibull_Generation


# np.set_printoptions(threshold=np.inf)
pd.options.display.max_rows = None
pd.options.display.max_columns = None

# # Step 2: Prepare the Data
# Prepare Demand Data
df_DE = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
df_DQ = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")

periods = len(df_DE)
print("periods = {}".format(periods))
demand_Elec_original = np.array(df_DE['E_plus'])
demand_Heat_original = np.array(df_DQ['Q_plus_1'])

# Prepare Exogenous Data
df_P_WT = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\wind_turbine.xls")
df_price = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_elec_price.xls")

P_WT_a_original = np.array(df_P_WT['Theoretical_Power_Curve (MW)'])
price_original = np.array(df_price['price/$'])

S=[]
for i in range(30):
    # mean=0, variance=0.1
    demand_Elec = Guass_Generation(demand_Elec_original,0,0.01)
    demand_Heat = Guass_Generation(demand_Heat_original,0,0.01)
    # mean=0, variance=0.00001
    price = Guass_Generation(price_original,0,0.000095)
    # shape=1, scale=0.1
    P_WT_a = Weibull_Generation(P_WT_a_original,1,0.1)

    scenario = np.vstack((demand_Elec,demand_Heat,price,P_WT_a))
    S.append(scenario)
# np.save('D:\\Nuts\CCGT\ADP\Stochastic\scenarios.npy',S)
# np.save('D:\\Nuts\CCGT\ADP\Stochastic\\test_scenarios.npy',S)
# np.save('D:\\Nuts\CCGT\ADP\Stochastic\\test_scenarios_1.npy',S)
np.save('D:\\Nuts\CCGT\ADP\Stochastic\\robustness_test.npy',S)