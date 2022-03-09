

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pylab import rcParams


rcParams['figure.figsize'] = 11, 5
pd.options.display.max_rows = None
pd.options.display.max_columns = None

# slopes 数组存储并初始化
slopes = np.zeros((96,5))

# 产生随机数 stochastic
import math
from random import gauss

my_mean = 0
load_variance = 0.1
# wind_variance = 0.2
price_variance = 0.00001


# Prepare Demand Data
df_DE = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
df_DQ = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")
periods = len(df_DE)
demand_Elec = np.array(df_DE['E_plus'])
demand_Heat = np.array(df_DQ['Q_plus_1'])

# Prepare Exogenous Data
df_P_WT = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\wind_turbine.xls")
df_price = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_elec_price.xls")
P_WT_a = np.array(df_P_WT['Theoretical_Power_Curve (MW)'])
price = np.array(df_price['price/$'])

# 加上不确定性
# 可视化，调整随机数区间
# (!) Demand Elec & Heat
fig,ax = plt.subplots()
for epoch in range(1,301):
    E_load_random_numbers = []
    H_load_random_numbers = []
    for i in range(96):
        E_load_random_numbers.append(gauss(my_mean, math.sqrt(load_variance)))
        H_load_random_numbers.append(gauss(my_mean, math.sqrt(load_variance)))
    E_load_random_numbers = np.array(E_load_random_numbers)
    H_load_random_numbers = np.array(H_load_random_numbers)
    demand_Elec_random = np.sum([demand_Elec, E_load_random_numbers], axis=0)
    demand_Heat_random = np.sum([demand_Heat, H_load_random_numbers], axis=0)
    ax.plot(demand_Elec_random, linestyle = '--', color = 'gray')
    ax.plot(demand_Heat_random, linestyle = '--', color = 'gray')

ax.plot(demand_Elec, linewidth = 3, color='blue', label='Power Demand')
ax.plot(demand_Heat, linewidth = 3, color='orange', label='Heat Demand')

ax.set_ylabel("Demand(MW)", fontsize = 15,fontname='Times New Roman')
ax.set_xlabel("Time Periods (interval=15min)", fontsize = 15,fontname='Times New Roman')
ax.set_xticks(range(1,100,5))
ax.set_xlim(-1.5,100)

plt.grid(linestyle='--',axis='y')

plt.savefig('D:\\Nuts\CCGT\ADP\Stochastic\Stochastic_Demand.jpeg',dpi=500,bbox_inches='tight')

plt.show()

# (2) Wind Power
def Nweibull(a,size, scale):
    return (scale*np.random.weibull(a,size)-0.3)

fig,ax = plt.subplots()
a = 1
scale = 0.1
for epoch in range(1,301):
    wind_random_numbers = Nweibull(a,96,scale)
    P_WT_a_random = np.sum([P_WT_a,wind_random_numbers], axis=0)
    ax.plot(P_WT_a_random, linestyle = '--', color = 'gray')
ax.plot(P_WT_a,linewidth = 3, color='limegreen')

ax.set_ylabel("Power(MW)", fontsize = 15,fontname='Times New Roman')
ax.set_xlabel("Time Periods (interval=15min)", fontsize = 15,fontname='Times New Roman')
ax.set_xticks(range(1,100,5))
ax.set_xlim(-1.5,100)

plt.grid(linestyle='--',axis='y')

plt.savefig('D:\\Nuts\CCGT\ADP\Stochastic\Stochastic_WindPower.jpeg',dpi=500,bbox_inches='tight')

plt.show()

# (3) Grid Price
fig,ax = plt.subplots()
x = np.arange(1,97)
for epoch in range(1,301):
    price_random_numbers = []
    for i in range(96):
        price_random_numbers.append(gauss(my_mean, math.sqrt(price_variance)))
    price_random_numbers = np.array(price_random_numbers)
    price_random = np.sum([price, price_random_numbers], axis=0)
    ax.step(x, price_random, linestyle = '--', color = 'gray')
ax.step(x, price, linewidth = 3, color='black')

ax.set_ylabel("Price($/kWh)", fontsize = 15,fontname='Times New Roman')
ax.set_xlabel("Time Periods (interval=15min)", fontsize = 15,fontname='Times New Roman')
ax.set_xlim(-5,100)

plt.grid(linestyle='--')

plt.savefig('D:\\Nuts\CCGT\ADP\Stochastic\Stochastic_ElectricPrice.jpeg',dpi=500,bbox_inches='tight')

plt.show()


