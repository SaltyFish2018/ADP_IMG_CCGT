
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pylab import rcParams

rcParams['figure.figsize'] = 11, 5
pd.options.display.max_rows = None
pd.options.display.max_columns = None
# # Step 2: Prepare the Data

# Prepare parameters

types = ("P_FC", "Q_GB", "g_CCGT", "P_wcur", "P_cur", "Q_cur")

adjustable_parameters_data = {"param1": [65, 300, 1460, 200, 150, 350]}

df_types = DataFrame(adjustable_parameters_data, index=types)


# Prepare Demand Data
# df_DE = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
# df_DQ = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")
df_DE = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
df_DQ = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")
periods = len(df_DE)
demand_Elec = Series(df_DE['E_plus'])
# demand_Elec = Series(df_DE['Edemand(MW)'])
# demand_Heat = Series(df_DQ['Qdemand(MW)'])
demand_Heat = Series(df_DQ['Q_plus_1'])
demand_Elec.index = range(1, periods + 1)
demand_Heat.index = range(1, periods + 1)

# Prepare Exogenous Data
df_P_WT = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\wind_turbine.xls")
df_price = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_elec_price.xls")
# df_P_WT = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\wind_turbine.xls")
P_WT_a = Series(df_P_WT['Theoretical_Power_Curve (MW)'])
P_WT_a.index = range(1, periods + 1)

# df_price = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_elec_price.xls")
price = Series(df_price['price/$'])
price.index = range(1, periods + 1)

# Create DataFrame
intervals = range(1, periods + 1)
Intervals =  range(1, 18*periods+1)
df_decision_vars = DataFrame({'P_FC': "", 'g_CCGT': "", 'P_GRID':"", 'P_c': "", 'u_c': "", 'P_d': "", 'u_d': "", 'P_wcur':"",
                              'P_cur': "", 'Q_GB': "", 'Q_HP': "", 'Q_cur': "", 'SOC': ""},index=intervals)
df_GRID_PRICE = DataFrame({'GRID_PRICE': ""},index=intervals)
df_P_CCGT = DataFrame({'P_CCGT': ""},index=intervals)
df_Q_CCGT = DataFrame({'Q_CCGT': ""},index=Intervals)
df_step_cost = DataFrame({'step_cost':""},index=range(1,94))

# Set index names
df_decision_vars.index.names = ['intervals']
df_GRID_PRICE.index.names = ['intervals']
df_P_CCGT.index.names = ['intervals']
df_Q_CCGT.index.names = ['Intervals']
df_step_cost.index.names = ['intervals']

# print(df_decision_vars)

# # Step 3: Model Setup

#######------计时开始--------#####
import time
# start = time.perf_counter()

import gurobipy as gp
from gurobipy import GRB

p = 4
moving_size = range(1, p + 1)

# t0 = time.perf_counter()

env = gp.Env()
env.start()
strategy = gp.Model("deterministic MPC")

# print(time.perf_counter() - t0)

## Start with interval = 1
i = 1
window = range(i,i+p+1)

# Variable Definition
P_FC = {}
for m in moving_size:
    P_FC[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.8, ub=7, name='P_FC.{}'.format(m))

g_CCGT = {}
for m in moving_size:
    g_CCGT[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.9918, ub=3.306, name='g_CCGT.{}'.format(m))

P_GRID = {}
for m in moving_size:
    P_GRID[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=-6, ub=6, name='P_GRID.{}'.format(m))

P_c = {}
for m in moving_size:
    P_c[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name='P_c.{}'.format(m))

u_c = {}
for m in moving_size:
    u_c[m] = strategy.addVar(vtype=GRB.BINARY, name='u_c.{}'.format(m))

P_d = {}
for m in moving_size:
    P_d[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name='P_d.{}'.format(m))

u_d = {}
for m in moving_size:
    u_d[m] = strategy.addVar(vtype=GRB.BINARY, name='u_d.{}'.format(m))

P_wcur = {}
for m in moving_size:
    P_wcur[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name='P_wcur.{}'.format(m))

P_cur = {}
for m in moving_size:
    P_cur[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='P_cur.{}'.format(m))

Q_GB = {}
for m in moving_size:
    Q_GB[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=15, name='Q_GB.{}'.format(m))

Q_HP = {}
for m in moving_size:
    Q_HP[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5, name='Q_HP.{}'.format(m))

Q_cur = {}
for m in moving_size:
    Q_cur[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='Q_cur.{}'.format(m))

SOC = {}
for m in moving_size:
    SOC[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='SOC.{}'.format(m))


Intervals_CCGT = range(1, 18*p+1)
Q_CCGT = {}
for Interval in Intervals_CCGT:
    Q_CCGT[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name='Q_CCGT.{}'.format(Interval))

strategy.update()


# Create decision variables DataFrame
df_median_decision = DataFrame(
    {'P_FC': P_FC, 'g_CCGT': g_CCGT, 'P_GRID': P_GRID, 'P_c': P_c, 'u_c': u_c, 'P_d': P_d, 'u_d': u_d, 'P_wcur': P_wcur,
     'P_cur': P_cur, 'Q_GB': Q_GB, 'Q_HP': Q_HP, 'Q_cur': Q_cur, 'SOC': SOC})

df_median_CCGT = DataFrame({'Q_CCGT': Q_CCGT})

# Set index names
df_median_decision.index.names = ['intervals_median']

df_median_CCGT.index.names = ['Intervals_median']

## Constraints
# Demand Constraints
# 1) Q_CCGT
for Interval in Intervals_CCGT:
    if Interval >= 5 and Interval <= 18:
        if Interval == 5:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4]-(0.2087 * df_median_decision.loc[1].g_CCGT) == 0)
        elif Interval == 6:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631) * df_median_decision.loc[1].g_CCGT) == 0)
        elif Interval == 7:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[1].g_CCGT) == 0)
        else:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[1].g_CCGT) == 0)
    elif Interval >= 19:
        if 0 < Interval % 18 <= 4:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *Q_CCGT[Interval - 3] \
                - 0.257 * Q_CCGT[Interval - 4]\
                - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[Interval // 18].g_CCGT == 0)
        elif Interval % 18 == 5:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - (0.2087 * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (0.0631 + 0.3656 + 0.4031) *
                   df_median_decision.loc[Interval // 18].g_CCGT) == 0)
        elif Interval % 18 == 6:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631) * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (0.3656 + 0.4031) *
                   df_median_decision.loc[Interval // 18].g_CCGT) == 0)
        elif Interval % 18 == 7:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[Interval // 18 + 1].g_CCGT + 0.4031 *
                   df_median_decision.loc[Interval // 18].g_CCGT) == 0)
        elif Interval % 18 >= 8:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4]\
                - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[Interval // 18 + 1].g_CCGT == 0)
        elif Interval % 18 == 0:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[Interval // 18].g_CCGT == 0)

strategy.update()



for Interval in Intervals_CCGT:
    if Interval >=2:
        strategy.addConstr(Q_CCGT[Interval] - Q_CCGT[Interval - 1] <= 1.5)
        strategy.addConstr(Q_CCGT[Interval] - Q_CCGT[Interval - 1] >= -1.5)

strategy.update()

# 2) Equations
for m, ii in df_median_decision.groupby(level='intervals_median'):
    strategy.addConstr(gp.quicksum(ii.P_FC + (ii.g_CCGT * 16 - 10) + ii.P_GRID + ii.P_d * ii.u_d + ii.P_cur + (-ii.P_wcur) \
                                   - ii.P_c * ii.u_c - ii.Q_HP / 4.5) + P_WT_a[i+m-1] - demand_Elec[i+m-1] == 0)
    strategy.addConstr(gp.quicksum(ii.Q_GB + ii.Q_HP + ii.Q_cur) + df_median_CCGT.loc[18 * m].Q_CCGT - demand_Heat[i+m-1] == 0)

strategy.update()

# 3) Ramping constraints
for m in moving_size:
    if m >= 2:
        # print(df_decision_vars.loc[interval+1].g_CCGT-df_decision_vars.loc[interval].g_CCGT)
        strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m-1].g_CCGT <= 1.5)
        strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m-1].g_CCGT >= -1.5)

strategy.update()

# 4) Battery Constraints
for m, ii in df_median_decision.groupby(level='intervals_median'):
    strategy.addConstr(gp.quicksum(ii.u_c + ii.u_d) <= 1)
    strategy.addConstr(gp.quicksum(ii.P_c - ii.u_c * 3) <= 0)
    strategy.addConstr(gp.quicksum(ii.P_d - ii.u_d * 3) <= 0)

    if m ==1:
        strategy.addConstr(gp.quicksum(ii.SOC - 7.5 - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9)) == 0)
    else:
        strategy.addConstr(gp.quicksum(ii.SOC - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9))-df_median_decision.loc[m-1].SOC == 0)

strategy.update()

for m in moving_size:
    strategy.addConstr(df_median_decision.loc[m].SOC >= 1.5)
    strategy.addConstr(df_median_decision.loc[m].SOC <= 15)

strategy.update()

#  Curtailment Constraints
for m in moving_size:
    strategy.addConstr(df_median_decision.loc[m].P_wcur - P_WT_a[i+m-1] <= 0)
    strategy.addConstr(df_median_decision.loc[m].P_cur - demand_Elec[i+m-1] <= 0)
    strategy.addConstr(df_median_decision.loc[m].Q_cur - demand_Heat[i+m-1] <= 0)

strategy.update()

# Objective
median_cost_1 = 0.25 * (gp.quicksum((df_types.loc["P_FC"].param1 * df_median_decision.P_FC \
                     + df_types.loc["Q_GB"].param1 * df_median_decision.Q_GB \
                     + df_types.loc["g_CCGT"].param1 * df_median_decision.g_CCGT) \
                    + (df_types.loc["P_wcur"].param1 * df_median_decision.P_wcur \
                       + df_types.loc["P_cur"].param1 * df_median_decision.P_cur \
                       + df_types.loc["Q_cur"].param1 * df_median_decision.Q_cur)))
median_cost_3 = 0
for m in moving_size:
    median_cost_2 = 0.25* 1000 * price[i+m-1] * df_median_decision.loc[m].P_GRID
    median_cost_3 += median_cost_2

strategy.update()

median_cost = (median_cost_1 + median_cost_3)
strategy.update()



strategy.setObjective(median_cost, sense=GRB.MINIMIZE)
strategy.update()
strategy.optimize()



# Save first step action and Update system state
P_FC_sol = strategy.getAttr('x', P_FC)
df_decision_vars.loc[i, 'P_FC'] = P_FC_sol[1]

g_CCGT_sol = strategy.getAttr('x', g_CCGT)
df_decision_vars.loc[i, 'g_CCGT'] = g_CCGT_sol[1]

P_GRID_sol = strategy.getAttr('x', P_GRID)
df_decision_vars.loc[i, 'P_GRID'] = P_GRID_sol[1]

P_c_sol = strategy.getAttr('x', P_c)
df_decision_vars.loc[i, 'P_c'] = P_c_sol[1]

u_c_sol = strategy.getAttr('x', u_c)
df_decision_vars.loc[i, 'u_c'] = u_c_sol[1]

P_d_sol = strategy.getAttr('x', P_d)
df_decision_vars.loc[i, 'P_d'] = P_d_sol[1]

u_d_sol = strategy.getAttr('x', u_d)
df_decision_vars.loc[i, 'u_d'] = u_d_sol[1]

P_wcur_sol = strategy.getAttr('x', P_wcur)
df_decision_vars.loc[i, 'P_wcur'] = P_wcur_sol[1]

P_cur_sol = strategy.getAttr('x', P_cur)
df_decision_vars.loc[i, 'P_cur'] = P_cur_sol[1]

Q_GB_sol = strategy.getAttr('x', Q_GB)
df_decision_vars.loc[i, 'Q_GB'] = Q_GB_sol[1]

Q_HP_sol = strategy.getAttr('x', Q_HP)
df_decision_vars.loc[i, 'Q_HP'] = Q_HP_sol[1]

Q_cur_sol = strategy.getAttr('x', Q_cur)
df_decision_vars.loc[i, 'Q_cur'] = Q_cur_sol[1]

SOC_sol = strategy.getAttr('x', SOC)
df_decision_vars.loc[i, 'SOC'] = SOC_sol[1]

# print(df_decision_vars)

Q_CCGT_sol = strategy.getAttr('x', Q_CCGT)
for iii in range(1,19):
    df_Q_CCGT.loc[(i-1) * 18 + iii].Q_CCGT = Q_CCGT_sol[iii]
# print(df_Q_CCGT)

df_P_CCGT.loc[i].P_CCGT = g_CCGT_sol[1] * 16 - 10
df_GRID_PRICE.loc[i].GRID_PRICE = 1000 * price[i] * P_GRID_sol[1]
# print(df_GRID_PRICE)

# Calculate and Save step cost
df_step_cost.loc[i].step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * df_decision_vars.loc[i].P_FC \
                     + df_types.loc["Q_GB"].param1 * df_decision_vars.loc[i].Q_GB \
                     + df_types.loc["g_CCGT"].param1 * df_decision_vars.loc[i].g_CCGT) \
                    + (df_types.loc["P_wcur"].param1 * df_decision_vars.loc[i].P_wcur \
                       + df_types.loc["P_cur"].param1 * df_decision_vars.loc[i].P_cur \
                       + df_types.loc["Q_cur"].param1 * df_decision_vars.loc[i].Q_cur) \
                    + 1000 * price[i] * df_decision_vars.loc[i].P_GRID)
# print(df_step_cost)

df_median_decision = DataFrame(
    {'P_FC': P_FC_sol, 'g_CCGT': g_CCGT_sol, 'P_GRID': P_GRID_sol, 'P_c': P_c_sol, 'u_c': u_c_sol, 'P_d': P_d_sol, \
     'u_d': u_d_sol, 'P_wcur': P_wcur_sol,'P_cur': P_cur_sol, 'Q_GB': Q_GB_sol, 'Q_HP': Q_HP_sol, 'Q_cur': Q_cur_sol, 'SOC': SOC_sol})

df_median_CCGT = DataFrame({'Q_CCGT': Q_CCGT_sol})

# print(df_median_decision)

# time periods 2--92
for i in range(2, periods-p+2):

    env = gp.Env()
    env.start()
    strategy = gp.Model("deterministic MPC%g"%i)

    # Variable Definition
    P_FC = {}
    for m in moving_size:
        P_FC[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.8, ub=7, name='P_FC.{}'.format(m))

    g_CCGT = {}
    for m in moving_size:
        g_CCGT[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.9918, ub=3.306, name='g_CCGT.{}'.format(m))

    P_GRID = {}
    for m in moving_size:
        P_GRID[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=-6, ub=6, name='P_GRID.{}'.format(m))

    P_c = {}
    for m in moving_size:
        P_c[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name='P_c.{}'.format(m))

    u_c = {}
    for m in moving_size:
        u_c[m] = strategy.addVar(vtype=GRB.BINARY, name='u_c.{}'.format(m))

    P_d = {}
    for m in moving_size:
        P_d[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name='P_d.{}'.format(m))

    u_d = {}
    for m in moving_size:
        u_d[m] = strategy.addVar(vtype=GRB.BINARY, name='u_d.{}'.format(m))

    P_wcur = {}
    for m in moving_size:
        P_wcur[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name='P_wcur.{}'.format(m))

    P_cur = {}
    for m in moving_size:
        P_cur[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='P_cur.{}'.format(m))

    Q_GB = {}
    for m in moving_size:
        Q_GB[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=15, name='Q_GB.{}'.format(m))

    Q_HP = {}
    for m in moving_size:
        Q_HP[m] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5, name='Q_HP.{}'.format(m))

    Q_cur = {}
    for m in moving_size:
        Q_cur[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='Q_cur.{}'.format(m))

    SOC = {}
    for m in moving_size:
        SOC[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='SOC.{}'.format(m))

    Intervals_CCGT = range(1, 18 * p + 1)
    Q_CCGT = {}
    for Interval in Intervals_CCGT:
        Q_CCGT[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name='Q_CCGT.{}'.format(Interval))

    strategy.update()

    # Create decision variables DataFrame
    df_median_decision = DataFrame(
        {'P_FC': P_FC, 'g_CCGT': g_CCGT, 'P_GRID': P_GRID, 'P_c': P_c, 'u_c': u_c, 'P_d': P_d, 'u_d': u_d,
         'P_wcur': P_wcur,
         'P_cur': P_cur, 'Q_GB': Q_GB, 'Q_HP': Q_HP, 'Q_cur': Q_cur, 'SOC': SOC})

    df_median_CCGT = DataFrame({'Q_CCGT': Q_CCGT})

    # Set index names
    df_median_decision.index.names = ['intervals_median']

    df_median_CCGT.index.names = ['Intervals_median']

    # Demand Constraints
    # 1) Q_CCGT
    for Interval in Intervals_CCGT:
        if Interval >=1 and Interval <=18:
            if Interval == 1:
                strategy.addConstr(Q_CCGT[Interval] - 1.63 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT + 0.6292 * df_Q_CCGT.loc[i * 18 - 19].Q_CCGT \
                                   + 0.3266 * df_Q_CCGT.loc[i * 18 - 20].Q_CCGT - 0.257 * df_Q_CCGT.loc[i * 18 - 21].Q_CCGT \
                                   - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT == 0)
            elif Interval == 2:
                strategy.addConstr(
                    Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT \
                    + 0.3266 * df_Q_CCGT.loc[i * 18 - 19].Q_CCGT - 0.257 * df_Q_CCGT.loc[i * 18 - 20].Q_CCGT \
                    - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT == 0)
            elif Interval == 3:
                strategy.addConstr(
                    Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                    + 0.3266 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT - 0.257 * df_Q_CCGT.loc[i * 18 - 19].Q_CCGT \
                    - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT == 0)
            elif Interval == 4:
                strategy.addConstr(
                    Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                        Interval - 3] \
                    - 0.257 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT \
                    - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT == 0)
            elif Interval == 5:
                strategy.addConstr(
                    Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                        Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                    - (0.2087 * df_median_decision.loc[1].g_CCGT + (0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT) == 0)
            elif Interval == 6:
                strategy.addConstr(
                    Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                        Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                    - ((0.2087 + 0.0631) * df_median_decision.loc[1].g_CCGT + (0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT) == 0)
            elif Interval == 7:
                strategy.addConstr(
                    Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                        Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                    - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[1].g_CCGT + 0.4031 * df_decision_vars.loc[i - 1].g_CCGT) == 0)
            elif Interval >= 8:
                strategy.addConstr(
                    Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                        Interval - 3] \
                    - 0.257 * Q_CCGT[Interval - 4] - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[1].g_CCGT == 0)
        elif Interval>=19:
            if 0 < Interval % 18 <= 4:
                strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                   + 0.3266 *Q_CCGT[Interval - 3] - 0.257 *Q_CCGT[Interval - 4] \
                    - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[Interval // 18].g_CCGT == 0)
            elif Interval % 18 == 5:
                strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                   + 0.3266 *Q_CCGT[Interval - 3] - 0.257 *Q_CCGT[Interval - 4] \
                    - (0.2087 * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (0.0631 + 0.3656 + 0.4031) *
                       df_median_decision.loc[Interval // 18].g_CCGT) == 0)
            elif Interval % 18 == 6:
                strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                   + 0.3266 *Q_CCGT[Interval - 3] - 0.257 *Q_CCGT[Interval - 4] \
                    - ((0.2087 + 0.0631) * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (0.3656 + 0.4031) *
                       df_median_decision.loc[Interval // 18].g_CCGT) == 0)
            elif Interval % 18 == 7:
                strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                   + 0.3266 *Q_CCGT[Interval - 3] - 0.257 *Q_CCGT[Interval - 4] \
                    - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[Interval // 18 + 1].g_CCGT + 0.4031 *
                       df_median_decision.loc[Interval // 18].g_CCGT) == 0)
            elif Interval % 18 >= 8:
                strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                   + 0.3266 *Q_CCGT[Interval - 3] - 0.257 *Q_CCGT[Interval - 4] \
                    - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[Interval // 18 + 1].g_CCGT == 0)
            elif Interval % 18 == 0:
                strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                   + 0.3266 *Q_CCGT[Interval - 3] - 0.257 *Q_CCGT[Interval - 4] \
                    - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[Interval // 18].g_CCGT == 0)

    strategy.update()

    for m, ii in df_median_decision.groupby(level='intervals_median'):
        strategy.addConstr(
            gp.quicksum(ii.P_FC + (ii.g_CCGT * 16 - 10) + ii.P_GRID + ii.P_d * ii.u_d + ii.P_cur + (-ii.P_wcur) \
                        - ii.P_c * ii.u_c - ii.Q_HP / 4.5) + P_WT_a[i + m - 1] - demand_Elec[i + m - 1] == 0)
        strategy.addConstr(gp.quicksum(ii.Q_GB + ii.Q_HP + ii.Q_cur) + df_median_CCGT.loc[18 * m].Q_CCGT - demand_Heat[i + m - 1] == 0)

    strategy.update()

    # 3) Ramping constraints
    for m in moving_size:
        if m >= 2:
            # print(df_decision_vars.loc[interval+1].g_CCGT-df_decision_vars.loc[interval].g_CCGT)
            strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m - 1].g_CCGT <= 1.5)
            strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m - 1].g_CCGT >= -1.5)

    strategy.update()

    # 4) Battery Constraints
    for m, ii in df_median_decision.groupby(level='intervals_median'):
        strategy.addConstr(gp.quicksum(ii.u_c + ii.u_d) <= 1)
        strategy.addConstr(gp.quicksum(ii.P_c - ii.u_c * 3) <= 0)
        strategy.addConstr(gp.quicksum(ii.P_d - ii.u_d * 3) <= 0)

        if m == 1:
            strategy.addConstr(gp.quicksum(ii.SOC - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9))- df_decision_vars.loc[i - 1].SOC == 0)
        else:
            strategy.addConstr(gp.quicksum(ii.SOC - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9)) - df_median_decision.loc[m - 1].SOC == 0)

    strategy.update()

    for m in moving_size:
        strategy.addConstr(df_median_decision.loc[m].SOC >= 1.5)
        strategy.addConstr(df_median_decision.loc[m].SOC <= 15)

    strategy.update()

    #  Curtailment Constraints
    for m in moving_size:
        strategy.addConstr(df_median_decision.loc[m].P_wcur - P_WT_a[i + m - 1] <= 0)
        strategy.addConstr(df_median_decision.loc[m].P_cur - demand_Elec[i + m - 1] <= 0)
        strategy.addConstr(df_median_decision.loc[m].Q_cur - demand_Heat[i + m - 1] <= 0)

    strategy.update()

    # Objective
    median_cost_1 = 0.25 * (gp.quicksum((df_types.loc["P_FC"].param1 * df_median_decision.P_FC \
                                         + df_types.loc["Q_GB"].param1 * df_median_decision.Q_GB \
                                         + df_types.loc["g_CCGT"].param1 * df_median_decision.g_CCGT) \
                                        + (df_types.loc["P_wcur"].param1 * df_median_decision.P_wcur \
                                           + df_types.loc["P_cur"].param1 * df_median_decision.P_cur \
                                           + df_types.loc["Q_cur"].param1 * df_median_decision.Q_cur)))
    median_cost_3 = 0
    for m in moving_size:
        median_cost_2 = 0.25 * 1000 * price[i + m - 1] * df_median_decision.loc[m].P_GRID
        median_cost_3 += median_cost_2

    strategy.update()

    median_cost = (median_cost_1 + median_cost_3)
    strategy.update()



    strategy.setObjective(median_cost, sense=GRB.MINIMIZE)
    strategy.update()
    strategy.optimize()



    # Save first step action and Update system state
    P_FC_sol = strategy.getAttr('x', P_FC)
    g_CCGT_sol = strategy.getAttr('x', g_CCGT)
    P_GRID_sol = strategy.getAttr('x', P_GRID)
    P_c_sol = strategy.getAttr('x', P_c)
    u_c_sol = strategy.getAttr('x', u_c)
    P_d_sol = strategy.getAttr('x', P_d)
    u_d_sol = strategy.getAttr('x', u_d)
    P_wcur_sol = strategy.getAttr('x', P_wcur)
    P_cur_sol = strategy.getAttr('x', P_cur)
    Q_GB_sol = strategy.getAttr('x', Q_GB)
    Q_HP_sol = strategy.getAttr('x', Q_HP)
    Q_cur_sol = strategy.getAttr('x', Q_cur)
    SOC_sol = strategy.getAttr('x', SOC)
    Q_CCGT_sol = strategy.getAttr('x', Q_CCGT)

    if i < (periods-p+1):
        df_decision_vars.loc[i, 'P_FC'] = P_FC_sol[1]
        df_decision_vars.loc[i, 'g_CCGT'] = g_CCGT_sol[1]
        df_decision_vars.loc[i, 'P_GRID'] = P_GRID_sol[1]
        df_decision_vars.loc[i, 'P_c'] = P_c_sol[1]
        df_decision_vars.loc[i, 'u_c'] = u_c_sol[1]
        df_decision_vars.loc[i, 'P_d'] = P_d_sol[1]
        df_decision_vars.loc[i, 'u_d'] = u_d_sol[1]
        df_decision_vars.loc[i, 'P_wcur'] = P_wcur_sol[1]
        df_decision_vars.loc[i, 'P_cur'] = P_cur_sol[1]
        df_decision_vars.loc[i, 'Q_GB'] = Q_GB_sol[1]
        df_decision_vars.loc[i, 'Q_HP'] = Q_HP_sol[1]
        df_decision_vars.loc[i, 'Q_cur'] = Q_cur_sol[1]
        df_decision_vars.loc[i, 'SOC'] = SOC_sol[1]

        for iii in range(1, 19):
            df_Q_CCGT.loc[(i - 1) * 18 + iii].Q_CCGT = Q_CCGT_sol[iii]
        # print(df_Q_CCGT)

        df_P_CCGT.loc[i].P_CCGT = g_CCGT_sol[1] * 16 - 10
        df_GRID_PRICE.loc[i].GRID_PRICE = 1000 * price[i] * P_GRID_sol[1]
        # Calculate and Save step cost
        df_step_cost.loc[i].step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * df_decision_vars.loc[i].P_FC \
                                                 + df_types.loc["Q_GB"].param1 * df_decision_vars.loc[i].Q_GB \
                                                 + df_types.loc["g_CCGT"].param1 * df_decision_vars.loc[i].g_CCGT) \
                                                + (df_types.loc["P_wcur"].param1 * df_decision_vars.loc[i].P_wcur \
                                                   + df_types.loc["P_cur"].param1 * df_decision_vars.loc[i].P_cur \
                                                   + df_types.loc["Q_cur"].param1 * df_decision_vars.loc[i].Q_cur) \
                                                + 1000 * price[i] * df_decision_vars.loc[i].P_GRID)

    else:
        for m in moving_size:
            df_decision_vars.loc[periods-p+m, 'P_FC'] = P_FC_sol[m]
            df_decision_vars.loc[periods-p+m, 'g_CCGT'] = g_CCGT_sol[m]
            df_decision_vars.loc[periods-p+m, 'P_GRID'] = P_GRID_sol[m]
            df_decision_vars.loc[periods-p+m, 'P_c'] = P_c_sol[m]
            df_decision_vars.loc[periods-p+m, 'u_c'] = u_c_sol[m]
            df_decision_vars.loc[periods-p+m, 'P_d'] = P_d_sol[m]
            df_decision_vars.loc[periods-p+m, 'u_d'] = u_d_sol[m]
            df_decision_vars.loc[periods-p+m, 'P_wcur'] = P_wcur_sol[m]
            df_decision_vars.loc[periods-p+m, 'P_cur'] = P_cur_sol[m]
            df_decision_vars.loc[periods-p+m, 'Q_GB'] = Q_GB_sol[m]
            df_decision_vars.loc[periods-p+m, 'Q_HP'] = Q_HP_sol[m]
            df_decision_vars.loc[periods-p+m, 'Q_cur'] = Q_cur_sol[m]
            df_decision_vars.loc[periods-p+m, 'SOC'] = SOC_sol[m]
            df_P_CCGT.loc[periods-p+m].P_CCGT = g_CCGT_sol[m] * 16 - 10
            df_GRID_PRICE.loc[periods-p+m].GRID_PRICE = 1000 * price[periods-p+m] * P_GRID_sol[m]

        for Interval in Intervals_CCGT:
            df_Q_CCGT.loc[(i - 1) * 18 + Interval].Q_CCGT = Q_CCGT_sol[Interval]
        # print(df_Q_CCGT)

        df_step_cost.loc[i].step_cost = strategy.objVal

print(df_step_cost['step_cost'].sum())

'''
# # Save Myopic/Greedy Result to files

outputpath = 'F:\CCGT\MPC\\1024MPC_result.csv'
df_decision_vars.to_csv(outputpath, sep=',', index=True, header=True)

outputpath1 = 'F:\CCGT\MPC\\1024MPC_Q_CCGT_all.csv'
df_Q_CCGT.to_csv(outputpath1, sep=',', index=True, header=True)

outputpath2 = 'F:\CCGT\MPC\\1024MPC_GRID_PRICE.csv'
df_GRID_PRICE.to_csv(outputpath2, sep=',', index=True, header=True)

outputpath3 = 'F:\CCGT\MPC\\1024MPC_step_cost.csv'
df_step_cost.to_csv(outputpath3, sep=',', index=True, header=True)

outputpath4 = 'F:\CCGT\MPC\\1024MPC_P_CCGT.csv'
df_P_CCGT.to_csv(outputpath4, sep=',', index=True, header=True)

'''


'''
##------- Plot Power assignment 并列柱状图 -------------
fig, ax1 = plt.subplots(1,1)            # 做1*1个子图，等价于 " fig, ax1 = plt.subplot() "，等价于 " fig, ax1 = plt.subplots() "
ax2 = ax1.twinx()                       # 让2个子图的x轴一样，同时创建副坐标轴。

intervals = range(1, periods + 1)
x = np.arange(len(intervals))
total_width,n=0.9,6
width = total_width/n
x = x-(total_width-width)/5

# width = 0.5

bar1 = ax1.bar(x, df_decision_vars.P_FC, width, label='P_FC')
bar2 = ax1.bar(x+width, df_P_CCGT.P_CCGT, width, label='P_CCGT')
bar3 = ax1.bar(x+2*width, df_decision_vars.P_GRID, width, label='P_GRID')
bar4 = ax1.bar(x+3*width, df_decision_vars.SOC, width, label='SOC')
bar5 = ax1.bar(x+4*width, P_WT_a, width, label='P_WT_a')
bar6 = ax1.bar(x+5*width, df_decision_vars.P_cur, width, label='P_cur')
bar7 = ax1.bar(x+4*width, df_decision_vars.P_wcur, width=width, label='P_wcur')
str = ax1.plot(intervals, demand_Elec, label='Elec_Demand')
str1 = ax2.plot(intervals, price, label='Elec_Price',color='black')


#设置x轴刻度
ax1.set_xticks(x)
ax1.set_xticklabels(intervals)
#设置x，y轴
ax1.set_ylabel("P(MW)", fontsize = 20)
ax2.set_ylabel("$/kWh",fontsize = 15)
ax1.set_xlabel("Time Periods", fontsize = 20)
ax2.set_ylim(-0.05,0.15)
# 显式图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
#显示标题
ax1.set_title('Myopic Policy Power Assignment')
# 显示网格
plt.grid(axis="y", linestyle='--')


##-------- Plot Heat assignment---------
fig, ax = plt.subplots()

x = np.arange(len(intervals))
total_width,n=0.8,4
width = total_width/n
x = x-(total_width-width)/3

Q_CCGT_sol = {}
for Interval in range(1, 18 * periods + 1):
    if Interval % 18 == 0:
        Q_CCGT_sol[Interval / 18] = df_Q_CCGT.loc[Interval].Q_CCGT

df_Q_CCGT_sol = DataFrame({'Q_CCGT': Q_CCGT_sol})

bar1 = ax.bar(x, df_decision_vars.Q_GB, width, label='Q_GB')
bar2 = ax.bar(x+width, df_decision_vars.Q_HP, width, label='Q_HP')
bar3 = ax.bar(x+2*width, df_decision_vars.Q_cur, width, label='Q_cur')
bar4 = ax.bar(x+3*width, df_Q_CCGT_sol.Q_CCGT, width, label='Q_CCGT')
str = ax.plot(intervals, demand_Heat, label='Heat_Demand')

#设置x轴刻度
ax.set_xticks(x)
ax.set_xticklabels(intervals)
#设置x，y轴
ax.set_ylabel("Q(MW)", fontsize = 20)
ax.set_xlabel("Time Periods", fontsize = 20)
# 显式图例
ax.legend()
#显示标题
ax1.set_title('Myopic Policy Heat Assignment')
# 显示网格
fig.tight_layout()
plt.grid(axis="y", linestyle='-.')
# plt.xticks(rotation=45)
# 3）xticks的旋转方面。例如上面的主副坐标轴的共x轴，要这样：
# ax.set_xticklabels(['str format labels'], rotation=80)
# 而这样的设置无效：plt.xticks(x, rotation=80)。

##-------- Plot Q_CCGT_All_sol ---------
fig, ax = plt.subplots()
ax.plot(range(1, 18 * periods + 1),df_Q_CCGT.Q_CCGT,label='Q_CCGT_all')

#设置x，y轴
ax.set_ylabel("Q_CCGT(MW)", fontsize = 20)
ax.set_xlabel("Time Periods", fontsize = 20)
# 显式图例
ax.legend()
#显示标题
ax1.set_title('Myopic Policy Power Assignment')
# 显示网格
plt.grid(axis="y", linestyle='-.')
fig.tight_layout()
'''




# ###------Stacked bar chart 堆叠柱形图-------###
# 1) Power
P_GRID = np.array(Series(df_decision_vars['P_GRID']))
P_CCGT = np.array(Series(df_P_CCGT['P_CCGT']))
P_FC = np.array(Series(df_decision_vars['P_FC']))
SOC = np.array(Series(df_decision_vars['SOC']))
P_wcur = np.array(Series(df_decision_vars['P_wcur']))
P_cur = np.array(Series(df_decision_vars['P_cur']))
P_WT_a = np.array(P_WT_a)

P_HP = np.array(Series(df_decision_vars['Q_HP']/4.5))
# print(P_HP)

fig, ax1 = plt.subplots(1,1)            # 做1*1个子图，等价于 " fig, ax1 = plt.subplot() "，等价于 " fig, ax1 = plt.subplots() "
# ax2 = ax1.twinx()                       # 让2个子图的x轴一样，同时创建副坐标轴。

intervals = range(1, periods + 1)
x = np.arange(len(intervals))
width = 0.5

bar3 = ax1.bar(x, P_GRID, width,label='P_GRID',facecolor='lightblue',edgecolor='blue')
bar2 = ax1.bar(x, P_CCGT, width, bottom=P_GRID, label='P_CCGT',facecolor='orange')
bar1 = ax1.bar(x, P_FC, width, bottom=(P_GRID+P_CCGT),label='P_FC')
bar4 = ax1.bar(x, P_WT_a, width, bottom=(P_GRID+P_CCGT+P_FC),label='P_WT_a',facecolor='lightcoral')
bar7 = ax1.bar(x, -1*P_wcur, width, bottom=(P_GRID+P_CCGT+P_FC+P_WT_a),label='P_wcur')
bar5 = ax1.bar(x, -1*P_HP, width, bottom=(P_GRID+P_CCGT+P_FC+P_WT_a),label='P2Q',color='brown')
bar6 = ax1.bar(x, P_cur, width,bottom=(P_GRID+P_CCGT+P_FC+P_WT_a-P_wcur) ,label='P_cur',facecolor='white',edgecolor='black',hatch='//')

str = ax1.plot(x, demand_Elec, label='Elec_Demand',color='b',linewidth=1.5)
# str1 = ax2.plot(x, price, label='Elec_Price',color='black')
str3 = ax1.plot(x, SOC,label='SOC',color='red',linestyle='-.')

#设置x轴刻度
ax1.set_xticks(x)
ax1.set_xticklabels(intervals)
#设置x，y轴
ax1.set_ylabel("P(MW)", fontsize = 15)
# ax2.set_ylabel("$/kWh",fontsize = 15)
ax1.set_xlabel("Time Periods", fontsize = 15)
# ax2.set_ylim(-0.05,0.15)
# 显式图例
ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
plt.grid(axis="y", linestyle='--')
# 画一条水平y=0的轴线
plt.axhline(y=0,linewidth=1, color='black')
# 显示标题
ax1.set_title('MPC Power Assignment',fontsize = 15,y = 1.05)

# 2) Heat
fig, ax = plt.subplots()

x = np.arange(len(intervals))

Q_GB = np.array(Series(df_decision_vars['Q_GB']))
Q_HP = np.array(Series(df_decision_vars['Q_HP']))
Q_CCGT_sol = {}
for Interval in range(1, 18 * periods + 1):
    if Interval % 18 == 0:
        Q_CCGT_sol[Interval / 18] = df_Q_CCGT.loc[Interval].Q_CCGT

Q_CCGT_sol = np.array(Series(Q_CCGT_sol))
Q_cur = np.array(Series(df_decision_vars['Q_cur']))

bar1 = ax.bar(x, Q_HP, width, label='Q_HP',color='gold')
bar2 = ax.bar(x, Q_GB, width, bottom = Q_HP,label='Q_GB',color='deepskyblue')
bar3 = ax.bar(x, Q_CCGT_sol, width, bottom = (Q_GB+Q_HP), label='Q_CCGT',color='salmon')
bar4 = ax.bar(x, Q_cur, width,bottom = (Q_GB+Q_HP+Q_CCGT_sol), label='Q_cur',facecolor='white',edgecolor='black',hatch='\\\\')
str = ax.plot(x, demand_Heat, label='Heat_Demand',color='b',linewidth=1.5)

# 设置x轴刻度
ax.set_xticks(x)
ax.set_xticklabels(intervals)
# 设置x，y轴
ax.set_ylabel("Q(MW)", fontsize = 15)
ax.set_xlabel("Time Periods", fontsize = 15)
# 显式图例
ax.legend()
# 显示标题
ax.set_title('MPC Heat Assignment',fontsize = 15, y = 1.05)
# 显示网格
fig.tight_layout()
plt.grid(axis="y", linestyle='-.')


plt.show()
