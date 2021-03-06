import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

rcParams['figure.figsize'] = 11, 5
pd.options.display.max_rows = None
pd.options.display.max_columns = None


# # Step 2: Prepare the Data

# Prepare parameters

types = ("P_FC", "Q_GB", "g_CCGT", "P_wcur", "P_cur", "Q_cur", "Q_waste", "SOC")

adjustable_parameters_data = {"param1": [100, 30, 200, 200, 350, 350, 200, 0]}

df_types = DataFrame(adjustable_parameters_data, index=types)



# Prepare Exogenous Data
test_scenarios = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\test_scenarios.npy')
demand_Elec = test_scenarios[0,0,:]
demand_Heat = test_scenarios[0,1,:]
# mean=0, variance=0.00001
price = test_scenarios[0,2,:]
# shape=1, scale=0.1
P_WT_a = test_scenarios[0,3,:]

periods = 96


# Create DataFrame
intervals = range(1, periods + 1)
Intervals = range(1, 18*periods+1)
df_decision_vars = DataFrame({'P_FC': "", 'g_CCGT': "", 'P_GRID':"", 'P_c': "", 'u_c': "", 'P_d': "", 'u_d': "", 'P_wcur':"",
                              'P_cur': "", 'Q_GB': "", 'Q_HP': "", 'Q_cur': "", 'SOC': "", 'Q_waste': ""},index=intervals)
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

test_Cost = []

import gurobipy as gp
from gurobipy import GRB



p = 4
moving_size = range(1, p + 1)

env = gp.Env()
env.start()
strategy = gp.Model("Stochastic MPC")

## Start with interval = 1
i = 1
window = range(i,i+p+1)

demand_Elec[i+p-4]-=1
demand_Heat[i+p-4]-=1


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

Q_waste = {}
for m in moving_size:
    Q_waste[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='Q_waste.{}'.format(m))

Intervals_CCGT = range(1, 18*p+1)
Q_CCGT = {}
for Interval in Intervals_CCGT:
    Q_CCGT[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name='Q_CCGT.{}'.format(Interval))

strategy.update()


# Create decision variables DataFrame
df_median_decision = DataFrame(
    {'P_FC': P_FC, 'g_CCGT': g_CCGT, 'P_GRID': P_GRID, 'P_c': P_c, 'u_c': u_c, 'P_d': P_d, 'u_d': u_d, 'P_wcur': P_wcur,
     'P_cur': P_cur, 'Q_GB': Q_GB, 'Q_HP': Q_HP, 'Q_cur': Q_cur, 'SOC': SOC, 'Q_waste':Q_waste})

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
                                   - ii.P_c * ii.u_c - ii.Q_HP / 4.5) + P_WT_a[i+m-2] - demand_Elec[i+m-2] == 0)
    strategy.addConstr(gp.quicksum(ii.Q_GB + ii.Q_HP + ii.Q_cur- ii.Q_waste) + df_median_CCGT.loc[18 * m].Q_CCGT - demand_Heat[i+m-2] == 0)

strategy.update()

# 3) Ramping constraints
for m in moving_size:
    if m >= 2:
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
    strategy.addConstr(df_median_decision.loc[m].P_wcur - P_WT_a[i+m-2] <= 0)
    strategy.addConstr(df_median_decision.loc[m].P_cur - demand_Elec[i+m-2] <= 0)
    strategy.addConstr(df_median_decision.loc[m].Q_cur - demand_Heat[i+m-2] <= 0)
    strategy.addConstr(df_median_decision.loc[m].Q_waste - demand_Heat[i+m-2] <= 0)

strategy.update()

# Objective
median_cost_1 = 0.25 * (gp.quicksum((df_types.loc["P_FC"].param1 * df_median_decision.P_FC \
                     + df_types.loc["Q_GB"].param1 * df_median_decision.Q_GB \
                     + df_types.loc["g_CCGT"].param1 * df_median_decision.g_CCGT) \
                    + (df_types.loc["P_wcur"].param1 * df_median_decision.P_wcur \
                       + df_types.loc["P_cur"].param1 * df_median_decision.P_cur \
                       + df_types.loc["Q_cur"].param1 * df_median_decision.Q_cur)\
                    + df_types.loc["Q_waste"].param1 * df_median_decision.Q_waste\
                    + df_types.loc["SOC"].param1 * df_median_decision.SOC))
median_cost_3 = 0
for m in moving_size:
    median_cost_2 = 0.25* 1000 * price[i+m-2] * df_median_decision.loc[m].P_GRID
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

Q_waste_sol = strategy.getAttr('x', Q_waste)
df_decision_vars.loc[i, 'Q_waste'] = Q_waste_sol[1]

# print(df_decision_vars)

Q_CCGT_sol = strategy.getAttr('x', Q_CCGT)
for iii in range(1,19):
    df_Q_CCGT.loc[(i-1) * 18 + iii].Q_CCGT = Q_CCGT_sol[iii]
# print(df_Q_CCGT)

df_P_CCGT.loc[i].P_CCGT = g_CCGT_sol[1] * 16 - 10
df_GRID_PRICE.loc[i].GRID_PRICE = 1000 * price[i-1] * P_GRID_sol[1]
# print(df_GRID_PRICE)

# Calculate and Save step cost
df_step_cost.loc[i].step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * df_decision_vars.loc[i].P_FC \
                     + df_types.loc["Q_GB"].param1 * df_decision_vars.loc[i].Q_GB \
                     + df_types.loc["g_CCGT"].param1 * df_decision_vars.loc[i].g_CCGT) \
                    + (df_types.loc["P_wcur"].param1 * df_decision_vars.loc[i].P_wcur \
                       + df_types.loc["P_cur"].param1 * df_decision_vars.loc[i].P_cur \
                       + df_types.loc["Q_cur"].param1 * df_decision_vars.loc[i].Q_cur)\
                    + df_types.loc["Q_waste"].param1 * df_decision_vars.loc[i].Q_waste\
                    + df_types.loc["SOC"].param1 * df_decision_vars.loc[i].SOC\
                    + 1000 * price[i-1] * df_decision_vars.loc[i].P_GRID)
# print(df_step_cost)

df_median_decision = DataFrame(
    {'P_FC': P_FC_sol, 'g_CCGT': g_CCGT_sol, 'P_GRID': P_GRID_sol, 'P_c': P_c_sol, 'u_c': u_c_sol, 'P_d': P_d_sol, \
     'u_d': u_d_sol, 'P_wcur': P_wcur_sol,'P_cur': P_cur_sol, 'Q_GB': Q_GB_sol, 'Q_HP': Q_HP_sol, 'Q_cur': Q_cur_sol, 'SOC': SOC_sol, 'Q_waste': Q_waste_sol})

df_median_CCGT = DataFrame({'Q_CCGT': Q_CCGT_sol})

demand_Elec[i+p-4]+=1
demand_Heat[i+p-4]+=1

# print(df_median_decision)

# time periods 2--92
for i in range(2, periods-p+2):
    env = gp.Env()
    env.start()
    strategy = gp.Model("deterministic MPC%g" % i)

    if i < (periods-p+1):
        demand_Elec[i + p - 4] -= 1
        demand_Heat[i + p - 4] -= 1

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

        Q_waste = {}
        for m in moving_size:
            Q_waste[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='Q_waste.{}'.format(m))

        Intervals_CCGT = range(1, 18 * p + 1)
        Q_CCGT = {}
        for Interval in Intervals_CCGT:
            Q_CCGT[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name='Q_CCGT.{}'.format(Interval))

        strategy.update()

        # Create decision variables DataFrame
        df_median_decision = DataFrame(
            {'P_FC': P_FC, 'g_CCGT': g_CCGT, 'P_GRID': P_GRID, 'P_c': P_c, 'u_c': u_c, 'P_d': P_d, 'u_d': u_d,
             'P_wcur': P_wcur,
             'P_cur': P_cur, 'Q_GB': Q_GB, 'Q_HP': Q_HP, 'Q_cur': Q_cur, 'SOC': SOC, 'Q_waste': Q_waste},
            index=moving_size)

        df_median_CCGT = DataFrame({'Q_CCGT': Q_CCGT}, index=Intervals_CCGT)

        # Set index names
        df_median_decision.index.names = ['intervals_median']

        df_median_CCGT.index.names = ['Intervals_median']

        # Demand Constraints
        # 1) Q_CCGT
        for Interval in Intervals_CCGT:
            if Interval >= 1 and Interval <= 18:
                if Interval == 1:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT + 0.6292 * df_Q_CCGT.loc[
                            i * 18 - 19].Q_CCGT \
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
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] \
                        - 0.257 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT \
                        - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT == 0)
                elif Interval == 5:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                        - (0.2087 * df_median_decision.loc[1].g_CCGT + (0.0631 + 0.3656 + 0.4031) *
                           df_decision_vars.loc[i - 1].g_CCGT) == 0)
                elif Interval == 6:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                        - ((0.2087 + 0.0631) * df_median_decision.loc[1].g_CCGT + (0.3656 + 0.4031) *
                           df_decision_vars.loc[i - 1].g_CCGT) == 0)
                elif Interval == 7:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                        - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[1].g_CCGT + 0.4031 *
                           df_decision_vars.loc[i - 1].g_CCGT) == 0)
                elif Interval >= 8:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] \
                        - 0.257 * Q_CCGT[Interval - 4] - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                            1].g_CCGT == 0)
            elif Interval >= 19:
                if 0 < Interval % 18 <= 4:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                                           Interval // 18].g_CCGT == 0)
                elif Interval % 18 == 5:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (
                                0.0631 + 0.3656 + 0.4031) *
                                          df_median_decision.loc[Interval // 18].g_CCGT) == 0)
                elif Interval % 18 == 6:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - ((0.2087 + 0.0631) * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (
                                0.3656 + 0.4031) *
                                          df_median_decision.loc[Interval // 18].g_CCGT) == 0)
                elif Interval % 18 == 7:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[
                        Interval // 18 + 1].g_CCGT + 0.4031 *
                                          df_median_decision.loc[Interval // 18].g_CCGT) == 0)
                elif Interval % 18 >= 8:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                                           Interval // 18 + 1].g_CCGT == 0)
                elif Interval % 18 == 0:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                                           Interval // 18].g_CCGT == 0)

        strategy.update()

        for m, ii in df_median_decision.groupby(level='intervals_median'):
            strategy.addConstr(
                gp.quicksum(ii.P_FC + (ii.g_CCGT * 16 - 10) + ii.P_GRID + ii.P_d * ii.u_d + ii.P_cur + (-ii.P_wcur) \
                            - ii.P_c * ii.u_c - ii.Q_HP / 4.5) + P_WT_a[i + m - 2] - demand_Elec[i + m - 2] == 0)
            strategy.addConstr(
                gp.quicksum(ii.Q_GB + ii.Q_HP + ii.Q_cur - ii.Q_waste) + df_median_CCGT.loc[18 * m].Q_CCGT -
                demand_Heat[i + m - 2] == 0)

        strategy.update()

        # 3) Ramping constraints
        for m in moving_size:
            if m >= 2:
                strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m - 1].g_CCGT <= 1.5)
                strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m - 1].g_CCGT >= -1.5)

        strategy.update()

        # 4) Battery Constraints
        for m, ii in df_median_decision.groupby(level='intervals_median'):
            strategy.addConstr(gp.quicksum(ii.u_c + ii.u_d) <= 1)
            strategy.addConstr(gp.quicksum(ii.P_c - ii.u_c * 3) <= 0)
            strategy.addConstr(gp.quicksum(ii.P_d - ii.u_d * 3) <= 0)

            if m == 1:
                strategy.addConstr(
                    gp.quicksum(ii.SOC - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9)) - df_decision_vars.loc[
                        i - 1].SOC == 0)
            else:
                strategy.addConstr(gp.quicksum(ii.SOC - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9)) -
                                   df_median_decision.loc[m - 1].SOC == 0)

        strategy.update()

        for m in moving_size:
            strategy.addConstr(df_median_decision.loc[m].SOC >= 1.5)
            strategy.addConstr(df_median_decision.loc[m].SOC <= 15)

        strategy.update()

        #  Curtailment Constraints
        for m in moving_size:
            strategy.addConstr(df_median_decision.loc[m].P_wcur - P_WT_a[i + m - 2] <= 0)
            strategy.addConstr(df_median_decision.loc[m].P_cur - demand_Elec[i + m - 2] <= 0)
            strategy.addConstr(df_median_decision.loc[m].Q_cur - demand_Heat[i + m - 2] <= 0)
            strategy.addConstr(df_median_decision.loc[m].Q_waste - demand_Heat[i + m - 2] <= 0)

        strategy.update()

        # Objective
        median_cost_1 = 0.25 * (gp.quicksum((df_types.loc["P_FC"].param1 * df_median_decision.P_FC \
                                             + df_types.loc["Q_GB"].param1 * df_median_decision.Q_GB \
                                             + df_types.loc["g_CCGT"].param1 * df_median_decision.g_CCGT) \
                                            + (df_types.loc["P_wcur"].param1 * df_median_decision.P_wcur \
                                               + df_types.loc["P_cur"].param1 * df_median_decision.P_cur \
                                               + df_types.loc["Q_cur"].param1 * df_median_decision.Q_cur)
                                            + df_types.loc["Q_waste"].param1 * df_median_decision.Q_waste
                                            + df_types.loc["SOC"].param1 * df_median_decision.SOC))
        median_cost_3 = 0
        for m in moving_size:
            median_cost_2 = 0.25 * 1000 * price[i + m - 2] * df_median_decision.loc[m].P_GRID
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
        Q_waste_sol = strategy.getAttr('x', Q_waste)

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
        df_decision_vars.loc[i, 'Q_waste'] = Q_waste_sol[1]

        for iii in range(1, 19):
            df_Q_CCGT.loc[(i - 1) * 18 + iii].Q_CCGT = Q_CCGT_sol[iii]
        # print(df_Q_CCGT)

        df_P_CCGT.loc[i].P_CCGT = g_CCGT_sol[1] * 16 - 10
        df_GRID_PRICE.loc[i].GRID_PRICE = 1000 * price[i-1] * P_GRID_sol[1]
        # Calculate and Save step cost
        df_step_cost.loc[i].step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * df_decision_vars.loc[i].P_FC \
                                                 + df_types.loc["Q_GB"].param1 * df_decision_vars.loc[i].Q_GB \
                                                 + df_types.loc["g_CCGT"].param1 * df_decision_vars.loc[i].g_CCGT) \
                                                + (df_types.loc["P_wcur"].param1 * df_decision_vars.loc[i].P_wcur \
                                                   + df_types.loc["P_cur"].param1 * df_decision_vars.loc[i].P_cur \
                                                   + df_types.loc["Q_cur"].param1 * df_decision_vars.loc[i].Q_cur)
                                                + df_types.loc["Q_waste"].param1 * df_decision_vars.loc[i].Q_waste
                                                + df_types.loc["SOC"].param1 * df_decision_vars.loc[i].SOC \
                                                + 1000 * price[i-1] * df_decision_vars.loc[i].P_GRID)
        demand_Elec[i + p - 4] += 1
        demand_Heat[i + p - 4] += 1

    else:
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

        Q_waste = {}
        for m in moving_size:
            Q_waste[m] = strategy.addVar(vtype=GRB.CONTINUOUS, name='Q_waste.{}'.format(m))

        Intervals_CCGT = range(1, 18 * p + 1)
        Q_CCGT = {}
        for Interval in Intervals_CCGT:
            Q_CCGT[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name='Q_CCGT.{}'.format(Interval))

        strategy.update()

        # Create decision variables DataFrame
        df_median_decision = DataFrame(
            {'P_FC': P_FC, 'g_CCGT': g_CCGT, 'P_GRID': P_GRID, 'P_c': P_c, 'u_c': u_c, 'P_d': P_d, 'u_d': u_d,
             'P_wcur': P_wcur,
             'P_cur': P_cur, 'Q_GB': Q_GB, 'Q_HP': Q_HP, 'Q_cur': Q_cur, 'SOC': SOC, 'Q_waste': Q_waste},
            index=moving_size)

        df_median_CCGT = DataFrame({'Q_CCGT': Q_CCGT}, index=Intervals_CCGT)

        # Set index names
        df_median_decision.index.names = ['intervals_median']

        df_median_CCGT.index.names = ['Intervals_median']

        # Demand Constraints
        # 1) Q_CCGT
        for Interval in Intervals_CCGT:
            if Interval >= 1 and Interval <= 18:
                if Interval == 1:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT + 0.6292 * df_Q_CCGT.loc[
                            i * 18 - 19].Q_CCGT \
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
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] \
                        - 0.257 * df_Q_CCGT.loc[i * 18 - 18].Q_CCGT \
                        - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[i - 1].g_CCGT == 0)
                elif Interval == 5:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                        - (0.2087 * df_median_decision.loc[1].g_CCGT + (0.0631 + 0.3656 + 0.4031) *
                           df_decision_vars.loc[i - 1].g_CCGT) == 0)
                elif Interval == 6:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                        - ((0.2087 + 0.0631) * df_median_decision.loc[1].g_CCGT + (0.3656 + 0.4031) *
                           df_decision_vars.loc[i - 1].g_CCGT) == 0)
                elif Interval == 7:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                        - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[1].g_CCGT + 0.4031 *
                           df_decision_vars.loc[i - 1].g_CCGT) == 0)
                elif Interval >= 8:
                    strategy.addConstr(
                        Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                        Q_CCGT[
                            Interval - 3] \
                        - 0.257 * Q_CCGT[Interval - 4] - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                            1].g_CCGT == 0)
            elif Interval >= 19:
                if 0 < Interval % 18 <= 4:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                                           Interval // 18].g_CCGT == 0)
                elif Interval % 18 == 5:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (
                                0.0631 + 0.3656 + 0.4031) *
                                          df_median_decision.loc[Interval // 18].g_CCGT) == 0)
                elif Interval % 18 == 6:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - ((0.2087 + 0.0631) * df_median_decision.loc[Interval // 18 + 1].g_CCGT + (
                                0.3656 + 0.4031) *
                                          df_median_decision.loc[Interval // 18].g_CCGT) == 0)
                elif Interval % 18 == 7:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - ((0.2087 + 0.0631 + 0.3656) * df_median_decision.loc[
                        Interval // 18 + 1].g_CCGT + 0.4031 *
                                          df_median_decision.loc[Interval // 18].g_CCGT) == 0)
                elif Interval % 18 >= 8:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                                           Interval // 18 + 1].g_CCGT == 0)
                elif Interval % 18 == 0:
                    strategy.addConstr(Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                                       + 0.3266 * Q_CCGT[Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                                       - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_median_decision.loc[
                                           Interval // 18].g_CCGT == 0)

        strategy.update()

        for m, ii in df_median_decision.groupby(level='intervals_median'):
            strategy.addConstr(
                gp.quicksum(ii.P_FC + (ii.g_CCGT * 16 - 10) + ii.P_GRID + ii.P_d * ii.u_d + ii.P_cur + (-ii.P_wcur) \
                            - ii.P_c * ii.u_c - ii.Q_HP / 4.5) + P_WT_a[i + m - 2] - demand_Elec[i + m - 2] == 0)
            strategy.addConstr(
                gp.quicksum(ii.Q_GB + ii.Q_HP + ii.Q_cur - ii.Q_waste) + df_median_CCGT.loc[18 * m].Q_CCGT -
                demand_Heat[i + m - 2] == 0)

        strategy.update()

        # 3) Ramping constraints
        for m in moving_size:
            if m >= 2:
                strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m - 1].g_CCGT <= 1.5)
                strategy.addConstr(df_median_decision.loc[m].g_CCGT - df_median_decision.loc[m - 1].g_CCGT >= -1.5)

        strategy.update()

        # 4) Battery Constraints
        for m, ii in df_median_decision.groupby(level='intervals_median'):
            strategy.addConstr(gp.quicksum(ii.u_c + ii.u_d) <= 1)
            strategy.addConstr(gp.quicksum(ii.P_c - ii.u_c * 3) <= 0)
            strategy.addConstr(gp.quicksum(ii.P_d - ii.u_d * 3) <= 0)

            if m == 1:
                strategy.addConstr(
                    gp.quicksum(ii.SOC - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9)) - df_decision_vars.loc[
                        i - 1].SOC == 0)
            else:
                strategy.addConstr(gp.quicksum(ii.SOC - 0.25 * (ii.P_c * ii.u_c * 0.9 - ii.P_d * ii.u_d / 0.9)) -
                                   df_median_decision.loc[m - 1].SOC == 0)

        strategy.update()

        for m in moving_size:
            strategy.addConstr(df_median_decision.loc[m].SOC >= 1.5)
            strategy.addConstr(df_median_decision.loc[m].SOC <= 15)

        strategy.update()

        #  Curtailment Constraints
        for m in moving_size:
            strategy.addConstr(df_median_decision.loc[m].P_wcur - P_WT_a[i + m - 2] <= 0)
            strategy.addConstr(df_median_decision.loc[m].P_cur - demand_Elec[i + m - 2] <= 0)
            strategy.addConstr(df_median_decision.loc[m].Q_cur - demand_Heat[i + m - 2] <= 0)
            strategy.addConstr(df_median_decision.loc[m].Q_waste - demand_Heat[i + m - 2] <= 0)

        strategy.update()

        # Objective
        median_cost_1 = 0.25 * (gp.quicksum((df_types.loc["P_FC"].param1 * df_median_decision.P_FC \
                                             + df_types.loc["Q_GB"].param1 * df_median_decision.Q_GB \
                                             + df_types.loc["g_CCGT"].param1 * df_median_decision.g_CCGT) \
                                            + (df_types.loc["P_wcur"].param1 * df_median_decision.P_wcur \
                                               + df_types.loc["P_cur"].param1 * df_median_decision.P_cur \
                                               + df_types.loc["Q_cur"].param1 * df_median_decision.Q_cur)
                                            + df_types.loc["Q_waste"].param1 * df_median_decision.Q_waste
                                            + df_types.loc["SOC"].param1 * df_median_decision.SOC))
        median_cost_3 = 0
        for m in moving_size:
            median_cost_2 = 0.25 * 1000 * price[i + m - 2] * df_median_decision.loc[m].P_GRID
            median_cost_3 += median_cost_2

        strategy.update()

        median_cost = (median_cost_1 + median_cost_3)
        strategy.update()

        strategy.setObjective(median_cost, sense=GRB.MINIMIZE)
        strategy.update()
        strategy.optimize()

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
            df_decision_vars.loc[periods-p+m, 'Q_waste'] = Q_waste_sol[m]
            df_P_CCGT.loc[periods-p+m].P_CCGT = g_CCGT_sol[m] * 16 - 10
            df_GRID_PRICE.loc[periods-p+m].GRID_PRICE = 1000 * price[periods-p+m-1] * P_GRID_sol[m]

        for Interval in Intervals_CCGT:
            df_Q_CCGT.loc[(i - 1) * 18 + Interval].Q_CCGT = Q_CCGT_sol[Interval]
        # print(df_Q_CCGT)

        df_step_cost.loc[i].step_cost = strategy.objVal



test_Cost.append(df_step_cost['step_cost'].sum())

print(test_Cost)

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

fig, ax1 = plt.subplots(1,1)            # ???1*1????????????????????? " fig, ax1 = plt.subplot() "???????????? " fig, ax1 = plt.subplots() "

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

str1 = ax1.plot(x, demand_Elec, label= 'Elec_Demand_random',color='b', linestyle='--',linewidth=1.5)
str2 = ax1.plot(x, SOC,label='SOC',color='red',linestyle='-.')


#??????x?????????
ax1.set_xticks(x)
ax1.set_xticklabels(intervals, rotation = 50)
#??????x???y???
ax1.set_ylabel("P(MW)", fontname='Times New Roman', fontsize = 20)
ax1.set_xlabel("Time Periods(interval=15min)", fontname='Times New Roman', fontsize = 20)
# ????????????
ax1.legend(loc='upper left')
plt.grid(axis="y", linestyle='--')
# ???????????????y=0?????????
plt.axhline(y=0,linewidth=1, color='black')
# ????????????
ax1.set_title('MPC Power Assignment (k=4)', fontname='Times New Roman',fontsize = 20,y = 1.05)

plt.show()

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
Q_waste = np.array(Series(df_decision_vars['Q_waste']))

bar1 = ax.bar(x, Q_HP, width, label='Q_HP',color='gold')
bar2 = ax.bar(x, Q_GB, width, bottom = Q_HP,label='Q_GB',color='deepskyblue')
bar3 = ax.bar(x, Q_CCGT_sol, width, bottom = (Q_GB+Q_HP), label='Q_CCGT',color='salmon')
bar4 = ax.bar(x, Q_cur, width,bottom = (Q_GB+Q_HP+Q_CCGT_sol), label='Q_cur',facecolor='white',edgecolor='black',hatch='\\\\')
bar5 = ax.bar(x, -Q_waste, width,bottom = (Q_GB+Q_HP+Q_CCGT_sol+Q_cur), label='Q_waste',facecolor='white',edgecolor='black',hatch='//')

str = ax.plot(x, demand_Heat, label='Heat_Demand_random',color='b',linewidth=1.5,linestyle='--')


# ??????x?????????
ax.set_xticks(x)
ax.set_xticklabels(intervals, rotation=50)
# ??????x???y???
ax.set_ylabel("Q(MW)", fontname='Times New Roman', fontsize = 20)
ax.set_xlabel("Time Periods(interval=15min)", fontname='Times New Roman', fontsize = 20)
# ????????????
ax.legend()
# ????????????
ax.set_title('MPC Heat Assignment (k=4)',fontname='Times New Roman', fontsize = 20, y = 1.05)
# ????????????
fig.tight_layout()
plt.grid(axis="y", linestyle='-.')


plt.show()