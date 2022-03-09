
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
print("periods = {}".format(periods))
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
df_decision_vars = DataFrame({'P_FC': "", 'g_CCGT': "", 'P_GRID':"", 'P_c': "", 'u_c': "", 'P_d': "", 'u_d': "", 'P_wcur':"",
                              'P_cur': "", 'Q_GB': "", 'Q_HP': "", 'Q_cur': "", 'SOC': ""},index=range(1, periods + 1))
df_GRID_PRICE = DataFrame({'GRID_PRICE': ""},index=range(1, periods + 1))
df_P_CCGT = DataFrame({'P_CCGT': ""},index=range(1, periods + 1))
df_Q_CCGT = DataFrame({'Q_CCGT': ""},index=range(1, 18 * periods + 1))
df_step_cost = DataFrame({'step_cost':""},index=range(1, periods + 1))

# Set index names
df_decision_vars.index.names = ['intervals']
df_GRID_PRICE.index.names = ['intervals']
df_P_CCGT.index.names = ['intervals']
df_Q_CCGT.index.names = ['Intervals']
df_step_cost.index.names = ['intervals']

# print(df_decision_vars)

# # Step 3: Model Setup



import gurobipy as gp
from gurobipy import GRB
import time

# t0 = time.perf_counter()

env = gp.Env()
env.start()

# print('env')
# print(time.perf_counter() - t0)

strategy = gp.Model("deterministic myopic")

# Variable Definition

# t1 = time.perf_counter()

P_FC = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.8, ub=7, name="P_FC")

g_CCGT = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.9918, ub=3.306, name="g_CCGT")

P_GRID = strategy.addVar(vtype=GRB.CONTINUOUS, lb=-6, ub=6, name="P_GRID")

P_c = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name="P_c")

u_c = strategy.addVar(vtype=GRB.BINARY, name="u_c")

P_d = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name="P_d")

u_d = strategy.addVar(vtype=GRB.BINARY, name="u_d")

P_wcur = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_wcur")

P_cur = strategy.addVar(vtype=GRB.CONTINUOUS, name="P_cur")

Q_GB = strategy.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=15, name="Q_GB")

Q_HP = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5, name="Q_HP")

Q_cur = strategy.addVar(vtype=GRB.CONTINUOUS, name="Q_cur")

SOC = strategy.addVar(vtype=GRB.CONTINUOUS, name="SOC")



Intervals_CCGT = range(1, 19)
Q_CCGT = {}
for Interval in Intervals_CCGT:
    Q_CCGT[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name='Q_CCGT.{}'.format(Interval))

strategy.update()



###### Step-by-step myopic #####
## Start interval = 1
interval = 1
# Demand Constraints
# 1) Q_CCGT
for Interval in Intervals_CCGT:

    if Interval >= 5 and Interval <= 18:
        if Interval == 5:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - (0.2087 * g_CCGT) == 0)
        elif Interval == 6:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631) * g_CCGT) == 0)
        elif Interval == 7:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631 + 0.3656) * g_CCGT) == 0)
        else:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 *
                Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631 + 0.3656 + 0.4031) * g_CCGT) == 0)

strategy.update()

for Interval in Intervals_CCGT:
        if Interval >=2:
            strategy.addConstr(Q_CCGT[Interval] - Q_CCGT[Interval - 1] <= 1.5)
            strategy.addConstr(Q_CCGT[Interval] - Q_CCGT[Interval - 1] >= -1.5)

strategy.update()

# 2) Equations
print(demand_Elec[interval])
strategy.addConstr((P_FC+ (g_CCGT * 16 - 10) + P_GRID + P_d * u_d + P_cur + (-P_wcur) - P_c * u_c \
                   -Q_HP / 4.5) + P_WT_a[interval] - demand_Elec[interval] == 0)
strategy.addConstr((Q_GB + Q_HP + Q_cur) + Q_CCGT[18 * interval] - demand_Heat[interval] == 0)

strategy.update()

# Battery Constraints
strategy.addConstr(SOC - 7.5 - 0.25 * (P_c * u_c * 0.9 - P_d * u_d / 0.9) == 0)

strategy.addConstr(u_c + u_d <= 1)
strategy.addConstr(P_c - u_c * 3 <= 0)
strategy.addConstr(P_d - u_d * 3 <= 0)

strategy.addConstr(SOC >= 1.5)
strategy.addConstr(SOC <= 15)

strategy.update()

#  Curtailment Constraints
strategy.addConstr(P_wcur - P_WT_a[interval] <= 0)
strategy.addConstr(P_cur - demand_Elec[interval] <= 0)
strategy.addConstr(Q_cur - demand_Heat[interval] <= 0)

strategy.update()


# Objective
step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * P_FC \
                     + df_types.loc["Q_GB"].param1 * Q_GB \
                     + df_types.loc["g_CCGT"].param1 * g_CCGT) \
                    + (df_types.loc["P_wcur"].param1 * P_wcur \
                       + df_types.loc["P_cur"].param1 * P_cur \
                       + df_types.loc["Q_cur"].param1 * Q_cur) \
                    + 1000 * price[interval] * P_GRID)

strategy.update()

# print('var')
# print(time.perf_counter() - t1)

strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
strategy.update()
strategy.optimize()

# Save step cost and Update system state

# for v in strategy.getVars():
#     print(' %s %g' % (v.varName, v.x))
# print ('Obj: %g' % strategy.objVal )

# t3= time.perf_counter()

df_step_cost.loc[interval].step_cost=strategy.ObjVal
# print(df_step_cost)


for v in strategy.getVars():
    df_decision_vars.loc[interval,'%s'%v.varName] = v.x
# print(df_decision_vars)
# print(type(g_CCGT.x))

Q_CCGT_sol = strategy.getAttr('x', Q_CCGT)
for i in Intervals_CCGT:
    df_Q_CCGT.loc[(interval-1) * 18 + i].Q_CCGT = Q_CCGT_sol[i]
# print(df_Q_CCGT)

df_P_CCGT.loc[interval].P_CCGT = g_CCGT.x * 16 - 10
df_GRID_PRICE.loc[interval].GRID_PRICE = 1000 * price[interval] * P_GRID.x
# print(df_GRID_PRICE)

# print("save：",time.perf_counter()-t3)
# print("Time used:",time.perf_counter() - t0)



for interval in range(2, periods+1):

    env = gp.Env()
    env.start()
    strategy = gp.Model("deterministic myopic%g"%interval)

    P_FC = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.8, ub=7, name="P_FC")

    g_CCGT = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.9918, ub=3.306, name="g_CCGT")

    P_GRID = strategy.addVar(vtype=GRB.CONTINUOUS, lb=-6, ub=6, name="P_GRID")

    P_c = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name="P_c")

    u_c = strategy.addVar(vtype=GRB.BINARY, name="u_c")

    P_d = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name="P_d")

    u_d = strategy.addVar(vtype=GRB.BINARY, name="u_d")

    P_wcur = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_wcur")

    P_cur = strategy.addVar(vtype=GRB.CONTINUOUS, name="P_cur")

    Q_GB = strategy.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=15, name="Q_GB")

    Q_HP = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5, name="Q_HP")

    Q_cur = strategy.addVar(vtype=GRB.CONTINUOUS, name="Q_cur")

    SOC = strategy.addVar(vtype=GRB.CONTINUOUS, name="SOC")

    Intervals_CCGT = range(1, 19)
    Q_CCGT = {}
    for Interval in Intervals_CCGT:
        Q_CCGT[Interval] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name='Q_CCGT.{}'.format(Interval))

    strategy.update()

    # Demand Constraints
    # 1) Q_CCGT
    for Interval in Intervals_CCGT:
        if Interval == 1:
            strategy.addConstr(Q_CCGT[Interval] - 1.63 * df_Q_CCGT.loc[interval * 18 - 18].Q_CCGT + 0.6292 * df_Q_CCGT.loc[
                interval * 18 - 19].Q_CCGT \
                               + 0.3266 * df_Q_CCGT.loc[interval * 18 - 20].Q_CCGT - 0.257 * df_Q_CCGT.loc[
                                   interval * 18 - 21].Q_CCGT \
                               - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[interval - 1].g_CCGT == 0)
        elif Interval == 2:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * df_Q_CCGT.loc[interval * 18 - 18].Q_CCGT \
                + 0.3266 * df_Q_CCGT.loc[interval * 18 - 19].Q_CCGT - 0.257 * df_Q_CCGT.loc[interval * 18 - 20].Q_CCGT \
                - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[interval - 1].g_CCGT == 0)
        elif Interval == 3:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] \
                + 0.3266 * df_Q_CCGT.loc[interval * 18 - 18].Q_CCGT - 0.257 * df_Q_CCGT.loc[interval * 18 - 19].Q_CCGT \
                - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[interval - 1].g_CCGT == 0)
        elif Interval == 4:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] \
                - 0.257 * df_Q_CCGT.loc[interval * 18 - 18].Q_CCGT \
                - (0.2087 + 0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[interval - 1].g_CCGT == 0)
        elif Interval == 5:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - (0.2087 * g_CCGT + (0.0631 + 0.3656 + 0.4031) * df_decision_vars.loc[interval - 1].g_CCGT) == 0)
        elif Interval == 6:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631) * g_CCGT + (0.3656 + 0.4031) * df_decision_vars.loc[interval - 1].g_CCGT) == 0)
        elif Interval == 7:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] - 0.257 * Q_CCGT[Interval - 4] \
                - ((0.2087 + 0.0631 + 0.3656) * g_CCGT + 0.4031 * df_decision_vars.loc[interval - 1].g_CCGT) == 0)
        elif Interval >= 8:
            strategy.addConstr(
                Q_CCGT[Interval] - 1.63 * Q_CCGT[Interval - 1] + 0.6292 * Q_CCGT[Interval - 2] + 0.3266 * Q_CCGT[
                    Interval - 3] \
                - 0.257 * Q_CCGT[Interval - 4] - (0.2087 + 0.0631 + 0.3656 + 0.4031) * g_CCGT == 0)

    strategy.update()
    # 2) Equations
    strategy.addConstr((P_FC + (g_CCGT * 16 - 10) + P_GRID + P_d * u_d + P_cur + (-P_wcur) - P_c * u_c \
                        - Q_HP / 4.5) + P_WT_a[interval] - demand_Elec[interval] == 0)
    strategy.addConstr((Q_GB + Q_HP + Q_cur) + Q_CCGT[Interval] - demand_Heat[interval] == 0)

    strategy.update()
    # Ramping Constraints
    strategy.addConstr(g_CCGT - df_decision_vars.loc[interval - 1].g_CCGT <= 1.5)
    strategy.addConstr(g_CCGT - df_decision_vars.loc[interval - 1].g_CCGT >= -1.5)

    strategy.update()
    # Battery Constraints
    strategy.addConstr(SOC - df_decision_vars.loc[interval - 1].SOC - 0.25 * (P_c * u_c * 0.9 - P_d * u_d / 0.9) == 0)
    strategy.addConstr(u_c + u_d <= 1)
    strategy.addConstr(P_c - u_c * 3 <= 0)
    strategy.addConstr(P_d - u_d * 3 <= 0)
    strategy.addConstr(SOC >= 1.5)
    strategy.addConstr(SOC <= 15)

    strategy.update()
    #  Curtailment Constraints
    strategy.addConstr(P_wcur - P_WT_a[interval] <= 0)
    strategy.addConstr(P_cur - demand_Elec[interval] <= 0)
    strategy.addConstr(Q_cur - demand_Heat[interval] <= 0)

    strategy.update()

    # Objective
    step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * P_FC \
                         + df_types.loc["Q_GB"].param1 * Q_GB \
                         + df_types.loc["g_CCGT"].param1 * g_CCGT) \
                        + (df_types.loc["P_wcur"].param1 * P_wcur \
                           + df_types.loc["P_cur"].param1 * P_cur \
                           + df_types.loc["Q_cur"].param1 * Q_cur) \
                        + 1000 * price[interval] * P_GRID)

    strategy.update()

    strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
    strategy.update()
    strategy.optimize()
    df_step_cost.loc[interval].step_cost = strategy.ObjVal
    # print(df_step_cost)

    for v in strategy.getVars():
        df_decision_vars.loc[interval, '%s' % v.varName] = v.x
    # print(df_decision_vars)
    # print(type(g_CCGT.x))

    Q_CCGT_sol = strategy.getAttr('x', Q_CCGT)
    for i in Intervals_CCGT:
        df_Q_CCGT.loc[(interval-1) * 18 + i].Q_CCGT = Q_CCGT_sol[i]
    # print(df_Q_CCGT)

    df_P_CCGT.loc[interval].P_CCGT = g_CCGT.x * 16 - 10
    df_GRID_PRICE.loc[interval].GRID_PRICE = 1000 * price[interval] * P_GRID.x



# # Save Myopic/Greedy Result to files

# outputpath = 'F:\CCGT\Myopic\\1020Myopic_result.csv'
# df_decision_vars.to_csv(outputpath, sep=',', index=True, header=True)
#
# outputpath1 = 'F:\CCGT\Myopic\\1020Myopic_Q_CCGT_all.csv'
# df_Q_CCGT.to_csv(outputpath1, sep=',', index=True, header=True)
#
# outputpath2 = 'F:\CCGT\Myopic\\1020Myopic_GRID_PRICE.csv'
# df_GRID_PRICE.to_csv(outputpath2, sep=',', index=True, header=True)
#
# outputpath3 = 'F:\CCGT\Myopic\\1020Myopic_step_cost.csv'
# df_step_cost.to_csv(outputpath3, sep=',', index=True, header=True)
#
# outputpath4 = 'F:\CCGT\Myopic\\1020Myopic_P_CCGT.csv'
# df_P_CCGT.to_csv(outputpath4, sep=',', index=True, header=True)

print(df_step_cost['step_cost'].sum())

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
ax1.set_title('Myopic Policy Power Assignment',fontsize = 15,y = 1.05)

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
ax.set_title('Myopic Policy Heat Assignment',fontsize = 15, y = 1.05)
# 显示网格
fig.tight_layout()
plt.grid(axis="y", linestyle='-.')


plt.show()

