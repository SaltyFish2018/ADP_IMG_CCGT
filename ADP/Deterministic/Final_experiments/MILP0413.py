#!/usr/bin/env python
# coding: utf-8

# # Step 1: Basic Configuration

# In[1]:


import sys
import datetime
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pylab import rcParams
import matplotlib.ticker as ticker
import time

rcParams['figure.figsize'] = 11, 5
now_time = datetime.datetime.now().strftime('%Y-%m-%d')
print(now_time)

# # Step 2: Prepare the Data

# Prepare parameters

types = ("P_FC", "Q_GB", "g_CCGT", "P_wcur", "P_cur", "Q_cur", "Q_waste", "SOC")#, "SOC")+ df_types.loc["SOC"].param1 * df_decision_vars.SOC\

adjustable_parameters_data = {"param1": [100, 30, 200, 200, 350, 350, 200, 0]}
# adjustable_parameters_data = {"param1": [130, 30, 300, 150, 150, 150, 150, 0]}

df_types = DataFrame(adjustable_parameters_data, index=types)


# Prepare Demand Data
df_DE = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
df_DQ = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")
# df_DE = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
# df_DQ = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")
periods = len(df_DE)
print("periods = {}".format(periods))
demand_Elec = Series(df_DE['E_plus'])
# demand_Elec = Series(df_DE['Edemand(MW)'])
demand_Heat = Series(df_DQ['Q_plus_1'])
# demand_Heat = Series(df_DQ['Q_plus'])
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

'''
# Plot demand
fig = plt.figure()
plt.xticks(fontsize = 15, fontname = 'Times New Roman')
plt.yticks(fontsize = 15, fontname = 'Times New Roman')
plt.xlabel("Time", fontsize = 20, fontname = 'Times New Roman')
plt.ylabel("Demand", fontsize = 20, fontname = 'Times New Roman')
# plt.plot(demand_Elec)
# plt.plot(demand_Heat)
# smooth line
x = demand_Elec.index
xnew = np.linspace(1,96,500) #300 represents number of points to make between T.min and T.max
demand_Elec_smooth = make_interp_spline(x,demand_Elec)(xnew)
demand_Heat_smooth = make_interp_spline(x,demand_Heat)(xnew)
plt.plot(xnew,demand_Elec_smooth)
plt.plot(xnew,demand_Heat_smooth)

plt.show()
'''

# # Step 3: Model Setup

time_start =  time.perf_counter()

import gurobipy as gp
from gurobipy import GRB

env = gp.Env()
env.start()

strategy = gp.Model("deterministic milp")
# strategy.params.nonConvex = 2
# strategy.params.timelimit = 30*60
# strategy.params.MIPGapAbs = 0.01


# Variable Definition

intervals = range(1, periods + 1)

P_FC = {}
for interval in intervals:
    P_FC[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.8, ub=7, name='P_FC.{}'.format(interval))

g_CCGT = {}
for interval in intervals:
    g_CCGT[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.9918, ub=3.306, name='g_CCGT.{}'.format(interval))

P_GRID = {}
for interval in intervals:
    P_GRID[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=-6, ub=6, name='P_GRID.{}'.format(interval))

P_c = {}
for interval in intervals:
    P_c[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name='P_c.{}'.format(interval))

u_c = {}
for interval in intervals:
    u_c[interval] = strategy.addVar(vtype=GRB.BINARY, name='u_c.{}'.format(interval))

P_d = {}
for interval in intervals:
    P_d[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name='P_d.{}'.format(interval))

u_d = {}
for interval in intervals:
    u_d[interval] = strategy.addVar(vtype=GRB.BINARY, name='u_d.{}'.format(interval))

P_wcur = {}
for interval in intervals:
    P_wcur[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name='P_wcur.{}'.format(interval))

P_cur = {}
for interval in intervals:
    P_cur[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, name='P_cur.{}'.format(interval))

Q_GB = {}
for interval in intervals:
    Q_GB[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=15, name='Q_GB.{}'.format(interval))

Q_HP = {}
for interval in intervals:
    Q_HP[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5, name='Q_HP.{}'.format(interval))

Q_cur = {}
for interval in intervals:
    Q_cur[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, name='Q_cur.{}'.format(interval))

SOC = {}
for interval in intervals:
    SOC[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, name='SOC.{}'.format(interval))

Q_waste = {}
for interval in intervals:
    Q_waste[interval] = strategy.addVar(vtype=GRB.CONTINUOUS, name='Q_waste.{}'.format(interval))

Intervals_CCGT = range(1, 18 * periods + 1)
Q_X1 = {}
Q_X2 = {}
Q_X3 = {}
Q_X4 = {}
Q_X5 = {}
Q_X6 = {}
Q_X7 = {}

lower_bound = 0
for Interval in Intervals_CCGT:
    Q_X1[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, name='Q_X1.{}'.format(Interval))
    Q_X2[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, name='Q_X2.{}'.format(Interval))
    Q_X3[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, name='Q_X3.{}'.format(Interval))
    Q_X4[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, name='Q_X4.{}'.format(Interval))
    Q_X5[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, name='Q_X5.{}'.format(Interval))
    Q_X6[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, name='Q_X6.{}'.format(Interval))
    Q_X7[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, name='Q_X7.{}'.format(Interval))

strategy.update()

# Organize all decision variables in a DataFrame indexed by 'units' and 'periods'
df_decision_vars = DataFrame(
    {'P_FC': P_FC, 'g_CCGT': g_CCGT, 'P_GRID': P_GRID, 'P_c': P_c, 'u_c': u_c, 'P_d': P_d, 'u_d': u_d, 'P_wcur': P_wcur,
     'P_cur': P_cur, 'Q_GB': Q_GB, 'Q_HP': Q_HP, 'Q_cur': Q_cur, 'SOC': SOC, 'Q_waste':Q_waste})

df_Q_X = DataFrame({'Q_X1':Q_X1, 'Q_X2':Q_X2, 'Q_X3':Q_X3, 'Q_X4':Q_X4, 'Q_X5':Q_X5, 'Q_X6':Q_X6, 'Q_X7':Q_X7})
df_Q_CCGT = DataFrame({'Q_CCGT': ""},index=range(1,periods+1))

# Set index names
df_decision_vars.index.names = ['intervals']

df_Q_X.index.names = ['Intervals']
df_Q_CCGT.index.names = ['Intervals']



# Express the Constrains

# 1) Q_ccgt constrains

for Interval in Intervals_CCGT:
    if Interval >= 2:
        strategy.addConstr(Q_X1[Interval] - Q_X2[Interval - 1] == 0)
        strategy.addConstr(Q_X2[Interval] - Q_X3[Interval - 1] == 0)
        strategy.addConstr(Q_X3[Interval] - Q_X4[Interval - 1] == 0)
        strategy.addConstr(Q_X4[Interval] - Q_X5[Interval - 1] == 0)
        strategy.addConstr(Q_X5[Interval] - Q_X6[Interval - 1] == 0)
        strategy.addConstr(Q_X6[Interval] - Q_X7[Interval - 1] == 0)
        if Interval % 18 == 0:
            strategy.addConstr(Q_X7[Interval] - 0.257 * Q_X4[Interval - 1] + 0.3266 * Q_X5[Interval - 1] \
                               + 0.6292 * Q_X6[Interval - 1] - 1.63 * Q_X7[Interval - 1] - df_decision_vars.loc[
                                   Interval // 18 ].g_CCGT == 0)
        else:
            strategy.addConstr(Q_X7[Interval] - 0.257 * Q_X4[Interval - 1] + 0.3266 * Q_X5[Interval - 1] \
                           + 0.6292 * Q_X6[Interval - 1] - 1.63 * Q_X7[Interval - 1] - df_decision_vars.loc[Interval // 18 + 1].g_CCGT == 0)

strategy.update()

for Interval in Intervals_CCGT:
    strategy.addConstr(0.4031 * Q_X1[Interval] + 0.3656 * Q_X2[Interval]+ 0.06311 * Q_X3[Interval] + 0.2087 * Q_X4[Interval]<=50)
    strategy.addConstr(0.4031 * Q_X1[Interval] + 0.3656 * Q_X2[Interval] + 0.06311 * Q_X3[Interval] + 0.2087 * Q_X4[Interval]>=15)

    if Interval >=2:
        strategy.addConstr(0.4031*(Q_X1[Interval]-Q_X1[Interval-1]) + 0.3656*(Q_X2[Interval]-Q_X2[Interval-1]) \
                               + 0.06311*(Q_X3[Interval]-Q_X3[Interval-1]) + 0.2087*(Q_X4[Interval]-Q_X4[Interval-1]) <= 1.5)
        strategy.addConstr(0.4031*(Q_X1[Interval]-Q_X1[Interval-1]) + 0.3656*(Q_X2[Interval]-Q_X2[Interval-1]) \
                               + 0.06311*(Q_X3[Interval]-Q_X3[Interval-1]) + 0.2087*(Q_X4[Interval]-Q_X4[Interval-1]) >= -1.5)
strategy.update()

# 2) Demand Constraints
for interval, i in df_decision_vars.groupby(level='intervals'):
    strategy.addConstr(gp.quicksum(i.P_FC + (i.g_CCGT * 16 - 10) + i.P_GRID + i.P_d * i.u_d + i.P_cur + (
        -i.P_wcur) - i.P_c * i.u_c - i.Q_HP / 4.5) + P_WT_a[interval] - demand_Elec[interval] == 0)
for interval in intervals:
    strategy.addConstr( (Q_GB[interval] + Q_HP[interval] +  Q_cur[interval]) + 0.4031 * Q_X1[18*interval] \
                        + 0.3656 * Q_X2[18*interval]+ 0.06311 * Q_X3[18*interval] + 0.2087 * Q_X4[18*interval] \
                        +(-Q_waste[interval])- demand_Heat[interval] == 0)


strategy.update()

# 3) ramping constraints

for interval in intervals:
    if interval <= 95:
        # print(df_decision_vars.loc[interval+1].g_CCGT-df_decision_vars.loc[interval].g_CCGT)
        strategy.addConstr(df_decision_vars.loc[interval + 1].g_CCGT - df_decision_vars.loc[interval].g_CCGT <= 1.5)
        strategy.addConstr(df_decision_vars.loc[interval + 1].g_CCGT - df_decision_vars.loc[interval].g_CCGT >= -1.5)

strategy.update()

# 4) Battery constraints

for interval, i in df_decision_vars.groupby(level='intervals'):

    strategy.addConstr(gp.quicksum(i.u_c + i.u_d) <= 1)
    strategy.addConstr(gp.quicksum(i.P_c - i.u_c * 3) <= 0)
    strategy.addConstr(gp.quicksum(i.P_d - i.u_d * 3) <= 0)

    if interval == 1:
        strategy.addConstr(df_decision_vars.loc[interval].SOC - 7.5 \
                           - 0.25 * (df_decision_vars.loc[interval].P_c * df_decision_vars.loc[interval].u_c * 0.9 \
                                     - df_decision_vars.loc[interval].P_d * df_decision_vars.loc[interval].u_d / 0.9) == 0)
    if interval >= 2:
        strategy.addConstr(df_decision_vars.loc[interval].SOC - df_decision_vars.loc[interval - 1].SOC \
                           - 0.25 * (df_decision_vars.loc[interval].P_c * df_decision_vars.loc[interval].u_c * 0.9 \
                                     - df_decision_vars.loc[interval].P_d * df_decision_vars.loc[interval].u_d / 0.9) == 0)

    strategy.addConstr(df_decision_vars.loc[interval].SOC >= 1.5)
    strategy.addConstr(df_decision_vars.loc[interval].SOC <= 15)

strategy.update()

# 5) Curtailment constraints

for interval in intervals:
    strategy.addConstr(df_decision_vars.loc[interval].P_wcur - P_WT_a[interval] <= 0)
    strategy.addConstr(df_decision_vars.loc[interval].P_cur - demand_Elec[interval] <= 0)
    strategy.addConstr(df_decision_vars.loc[interval].Q_cur - demand_Heat[interval] <= 0)
    strategy.addConstr(df_decision_vars.loc[interval].Q_waste - demand_Heat[interval] <= 0)

strategy.update()

# Objective
total_cost = 0
for interval in intervals:
    step_cost = 0.25 * (df_types.loc["P_FC"].param1 * df_decision_vars.loc[interval,'P_FC']+
                        df_types.loc["Q_GB"].param1 * df_decision_vars.loc[interval,'Q_GB']+
                        df_types.loc["g_CCGT"].param1 * df_decision_vars.loc[interval,'g_CCGT']+
                        df_types.loc["P_wcur"].param1 * df_decision_vars.loc[interval,'P_wcur']+
                        df_types.loc["P_cur"].param1 * df_decision_vars.loc[interval,'P_cur']+
                        df_types.loc["Q_cur"].param1 * df_decision_vars.loc[interval,'Q_cur']+
                        df_types.loc["Q_waste"].param1 * df_decision_vars.loc[interval,'Q_waste']\
                        + 1000 * price[interval] * df_decision_vars.loc[interval,'P_GRID'])\
                + df_types.loc["SOC"].param1 * df_decision_vars.loc[interval,'SOC']

    total_cost += step_cost


strategy.update()

# Express the Objective

# #######------计时开始--------#####
# import time
#
# start = time.perf_counter()

# Objective
strategy.setObjective(total_cost, sense=GRB.MINIMIZE)

strategy.update()

# # Step 4: Solve with Decision Optimizaiton


# strategy.params.Method = 2


strategy.optimize()

# Save results
print(strategy.objVal)
time_end = time.perf_counter()-time_start
print(f'used time:{time_end}')


# elapsed = (time.perf_counter() - start)
# print("Time used:",elapsed)
# #######------计时结束--------#####



# save solution as dataframe, and write in pickle file
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
Q_waste_sol = strategy.getAttr('x',Q_waste)

Q_X1_sol = strategy.getAttr('x',Q_X1)
Q_X2_sol = strategy.getAttr('x',Q_X2)
Q_X3_sol = strategy.getAttr('x',Q_X3)
Q_X4_sol = strategy.getAttr('x',Q_X4)
Q_X5_sol = strategy.getAttr('x',Q_X5)
Q_X6_sol = strategy.getAttr('x',Q_X6)
Q_X7_sol = strategy.getAttr('x',Q_X7)



# ORIGINAL RESULT
df_decision_vars_sol = DataFrame({'P_FC': P_FC_sol, 'g_CCGT': g_CCGT_sol, 'P_GRID': P_GRID_sol, \
                                  'P_c': P_c_sol, 'u_c': u_c_sol, 'P_d': P_d_sol, 'u_d': u_d_sol, \
                                  'P_wcur': P_wcur_sol, 'P_cur': P_cur_sol, 'Q_GB': Q_GB_sol, 'Q_HP': Q_HP_sol, \
                                  'Q_cur': Q_cur_sol, 'SOC': SOC_sol, 'Q_waste': Q_waste_sol})



# GRID PRICE
df_GRID_PRICE_sol = DataFrame({'GRID_PRICE': ""},index=range(1, periods + 1))
df_GRID_PRICE_sol.index.names = ['intervals']
for interval in intervals:
    df_GRID_PRICE_sol.loc[interval].GRID_PRICE=0.25*1000*df_decision_vars_sol.loc[interval].P_GRID*price[interval]
# print(df_GRID_PRICE_sol)

# P_CCGT
P_CCGT_sol = {}
for interval in intervals:
    P_CCGT_sol[interval]=df_decision_vars_sol.loc[interval].g_CCGT*16-10
df_P_CCGT_sol = DataFrame({'P_CCGT':P_CCGT_sol})

# Q_CCGT
Q_CCGT_All_sol = {}
Q_CCGT_sol = {}
for Interval in Intervals_CCGT:
    Q_CCGT_All_sol[Interval] = 0.4031 * Q_X1_sol[Interval] + 0.3656 * Q_X2_sol[Interval]+ 0.06311 * Q_X3_sol[Interval] + 0.2087 * Q_X4_sol[Interval]
    if Interval % 18 == 0:
        Q_CCGT_sol[Interval / 18] = 0.4031 * Q_X1_sol[Interval] + 0.3656 * Q_X2_sol[Interval]+ 0.06311 * Q_X3_sol[Interval] + 0.2087 * Q_X4_sol[Interval]

df_Q_CCGT_All_sol = DataFrame({'Q_CCGT':Q_CCGT_All_sol})
df_Q_CCGT_sol = DataFrame({'Q_CCGT': Q_CCGT_sol})

# print(df_Q_CCGT_All_sol)
# print(df_Q_CCGT_sol)




# Set index names

df_decision_vars_sol.index.names = ['intervals']
df_Q_CCGT_sol.index.names = ['intervals']
df_P_CCGT_sol.index.names = ['intervals']
df_GRID_PRICE_sol.index.names = ['intervals']
df_Q_CCGT_All_sol.index.names = ['Intervals']
# print(df_Q_CCGT_sol)

# # Save MILP Result to files
# df_decision_vars_sol.head()
outputpath = 'D:\\Nuts\CCGT\ADP\Deterministic\Q_ccgt\\04MILP_result.csv'
df_decision_vars_sol.to_csv(outputpath, sep=',', index=True, header=True)
#
outputpath1 = 'D:\\Nuts\CCGT\ADP\Deterministic\Q_ccgt\\04MILP_Q_CCGT_all.csv'
df_Q_CCGT_All_sol.to_csv(outputpath1, sep=',', index=True, header=True)
#
# outputpath2 = 'D:\\Nuts\CCGT\MILP\\0405MILP_GRID_PRICE.csv'
# df_GRID_PRICE_sol.to_csv(outputpath2, sep=',', index=True, header=True)
#
# outputpath3 = 'D:\\Nuts\CCGT\MILP\\0405MILP_Q_CCGT.csv'
# df_Q_CCGT_sol.to_csv(outputpath3, sep=',', index=True, header=True)

'''
##------- Plot Power assignment 并列柱状图 -------------
fig, ax1 = plt.subplots(1,1)            # 做1*1个子图，等价于 " fig, ax1 = plt.subplot() "，等价于 " fig, ax1 = plt.subplots() "
ax2 = ax1.twinx()                       # 让2个子图的x轴一样，同时创建副坐标轴。

x = np.arange(len(intervals))
total_width,n=0.9,6
width = total_width/n
x = x-(total_width-width)/5

# width = 0.5

bar1 = ax1.bar(x, df_decision_vars_sol.P_FC, width, label='P_FC')
bar2 = ax1.bar(x+width, df_P_CCGT_sol.P_CCGT, width, label='P_CCGT')
bar3 = ax1.bar(x+2*width, df_decision_vars_sol.P_GRID, width, label='P_GRID')
bar4 = ax1.bar(x+3*width, df_decision_vars_sol.SOC, width, label='SOC')
bar5 = ax1.bar(x+4*width, P_WT_a, width, label='P_WT_a')
bar6 = ax1.bar(x+5*width, df_decision_vars_sol.P_cur, width, label='P_cur')
bar7 = ax1.bar(x+4*width, df_decision_vars_sol.P_wcur, width=width, label='P_wcur')
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
plt.grid(axis="y", linestyle='--')

##-------- Plot Heat assignment---------
fig, ax = plt.subplots()

x = np.arange(len(intervals))
total_width,n=0.8,4
width = total_width/n
x = x-(total_width-width)/3


bar1 = ax.bar(x, df_decision_vars_sol.Q_GB, width, label='Q_GB')
bar2 = ax.bar(x+width, df_decision_vars_sol.Q_HP, width, label='Q_HP')
bar3 = ax.bar(x+2*width, df_decision_vars_sol.Q_cur, width, label='Q_cur')
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
fig.tight_layout()
plt.grid(axis="y", linestyle='-.')
# plt.xticks(rotation=45)
# 3）xticks的旋转方面。例如上面的主副坐标轴的共x轴，要这样：
# ax.set_xticklabels(['str format labels'], rotation=80)
# 而这样的设置无效：plt.xticks(x, rotation=80)。

##-------- Plot Q_CCGT_All_sol ---------
fig, ax = plt.subplots()
ax.plot(Intervals_CCGT,df_Q_CCGT_All_sol.Q_CCGT,label='Q_CCGT_all')

#设置x，y轴
ax.set_ylabel("Q_CCGT(MW)", fontsize = 20)
ax.set_xlabel("Time Periods", fontsize = 20)
# 显式图例
ax.legend()
fig.tight_layout()

plt.grid(axis="y", linestyle='-.')
plt.show()
'''

# ###------Stacked bar chart 堆叠柱形图-------###
# 1) Power
P_GRID = np.array(Series(df_decision_vars_sol['P_GRID']))
P_CCGT = np.array(Series(df_P_CCGT_sol['P_CCGT']))
P_FC = np.array(Series(df_decision_vars_sol['P_FC']))
SOC = np.array(Series(df_decision_vars_sol['SOC']))
P_wcur = np.array(Series(df_decision_vars_sol['P_wcur']))
P_cur = np.array(Series(df_decision_vars_sol['P_cur']))
P_WT_a = np.array(P_WT_a)

P_HP = np.array(Series(df_decision_vars_sol['Q_HP']/4.5))
# print(P_HP)

fig, ax1 = plt.subplots(1,1)            # 做1*1个子图，等价于 " fig, ax1 = plt.subplot() "，等价于 " fig, ax1 = plt.subplots() "
# ax2 = ax1.twinx()                       # 让2个子图的x轴一样，同时创建副坐标轴。

intervals = range(1, periods + 1)
width = 0.5

bar3 = ax1.bar(intervals, P_GRID, width,label='P_GRID',facecolor='lightblue',edgecolor='blue')
bar2 = ax1.bar(intervals, P_CCGT, width, bottom=P_GRID, label='P_CCGT',facecolor='orange')
bar1 = ax1.bar(intervals, P_FC, width, bottom=(P_GRID+P_CCGT),label='P_FC')
bar4 = ax1.bar(intervals, P_WT_a, width, bottom=(P_GRID+P_CCGT+P_FC),label='P_WT_a',facecolor='lightcoral')
bar7 = ax1.bar(intervals, -1*P_wcur, width, bottom=(P_GRID+P_CCGT+P_FC+P_WT_a),label='P_wcur')
bar5 = ax1.bar(intervals, -1*P_HP, width, bottom=(P_GRID+P_CCGT+P_FC+P_WT_a),label='P2Q',color='brown')
bar6 = ax1.bar(intervals, P_cur, width,bottom=(P_GRID+P_CCGT+P_FC+P_WT_a-P_wcur) ,label='P_cur',facecolor='white',edgecolor='black',hatch='//')

str = ax1.plot(intervals, demand_Elec, label='Elec_Demand',color='b',linewidth=2)
# str1 = ax2.plot(x, price, label='Elec_Price',color='black')
str3 = ax1.plot(intervals, SOC, label='SOC', color='red',linestyle='--',linewidth=3)

#设置x轴刻度
# xticks = list(range(1,len(intervals),4))
# ax1.set_xticks(xticks)
# ax1.xlabels=[intervals[x] for x in xticks]
# ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=4))
ax1.set_xticks(intervals)
ax1.set_xticklabels(intervals, rotation=50)
ax1.set_xlim(-0.5,97)
#设置x，y轴
ax1.set_ylabel("P(MW)", fontsize = 20, fontname='Times New Roman')
# ax2.set_ylabel("$/kWh",fontsize = 20, fontname='Times New Roman')
ax1.set_xlabel("Time Periods (interval=15min)", fontname='Times New Roman',fontsize = 20)
# ax2.set_ylim(-0.05,0.15)
# 显式图例
ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
plt.grid(axis="y", linestyle='--')
# 画一条水平y=0的轴线
plt.axhline(y=-0.03238,linewidth=1, color='black')
# 显示标题
ax1.set_title('MILP Power Assignment',fontsize = 25, fontname='Times New Roman', y = 1.05)

# 2) Heat
fig, ax = plt.subplots()


Q_GB = np.array(Series(df_decision_vars_sol['Q_GB']))
Q_HP = np.array(Series(df_decision_vars_sol['Q_HP']))
Q_CCGT_sol = np.array(Series(df_Q_CCGT_sol['Q_CCGT']))
Q_cur = np.array(Series(df_decision_vars_sol['Q_cur']))
Q_waste = np.array(Series(df_decision_vars_sol['Q_waste']))

bar1 = ax.bar(intervals, Q_HP, width, label='Q_HP',color='gold')
bar2 = ax.bar(intervals, Q_GB, width, bottom = Q_HP,label='Q_GB',color='deepskyblue')
bar3 = ax.bar(intervals, Q_CCGT_sol, width, bottom = (Q_GB+Q_HP), label='Q_CCGT',color='salmon')
bar4 = ax.bar(intervals, Q_cur, width,bottom = (Q_GB+Q_HP+Q_CCGT_sol), label='Q_cur',facecolor='white',edgecolor='black',hatch='\\\\')
bar5 = ax.bar(intervals, -Q_waste, width,bottom = (Q_GB+Q_HP+Q_CCGT_sol+Q_cur), label='Q_waste',facecolor='white',edgecolor='black',hatch='//')
str = ax.plot(intervals, demand_Heat, label='Heat_Demand',color='b',linewidth=2)

# 设置x轴刻度
ax.set_xticks(intervals)
ax.set_xticklabels(intervals, rotation=50)
ax.set_xlim(-0.5,97)
# 设置x，y轴
ax.set_ylabel("Q(MW)", fontsize = 20, fontname='Times New Roman')
ax.set_xlabel("Time Periods (interval=15min)", fontsize = 20, fontname='Times New Roman')
# 显式图例
ax.legend()
# 显示标题
ax.set_title('MILP Heat Assignment', fontsize = 25, fontname='Times New Roman', y = 1.05)
# 显示网格
fig.tight_layout()
plt.grid(axis="y", linestyle='-.')

# 3) Q_CCGT
fig, ax = plt.subplots()
ax.plot(range(1, 18 * periods + 1),df_Q_CCGT_All_sol.Q_CCGT,label='Q_CCGT_all')

#设置x，y轴
ax.set_ylabel("Q_CCGT(MW)", fontsize = 20)
ax.set_xlabel("Time Periods", fontsize = 20)
# 显式图例
ax.legend()
# 显示网格
plt.grid(axis="y", linestyle='-.')
fig.tight_layout()
# # 显示标题
# ax.set_title('Q_CCGT_all',fontsize = 15, y = 1.05)

plt.show()