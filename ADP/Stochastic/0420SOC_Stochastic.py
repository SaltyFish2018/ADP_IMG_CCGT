

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams



# np.set_printoptions(threshold=np.inf)

rcParams['figure.figsize'] = 11, 5
pd.options.display.max_rows = None
pd.options.display.max_columns = None


Nt = 5      #记录分段数Nt
slopes = np.zeros((96,5))

Cost = []    #第n次迭代中的目标函数

# # Step 2: Prepare the Data

# Prepare parameters

# types = ("P_FC", "Q_GB", "g_CCGT", "P_wcur", "P_cur", "Q_cur", "Q_waste", "SOC")

types = np.array((100, 30, 200, 200, 350, 350, 200, 0))


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

# # Create DataFrame
# df_decision_vars = DataFrame({'P_FC': "", 'g_CCGT': "", 'P_GRID':"", 'P_c': "", 'u_c': "", 'P_d': "", 'u_d': "", 'P_wcur':"",
#                               'P_cur': "", 'Q_GB': "", 'Q_HP': "", 'Q_cur': "", 'SOC': "", 'Q_waste':""},index=range(1, periods + 1))

decisions = np.zeros((96,17))

# df_Q_CCGT = DataFrame({'Q_CCGT': ""},index=range(1, 18*periods + 1))
# df_Q_X1 = DataFrame({'Q_X1': ""},index=range(1, 18*periods + 1))
# df_Q_X2 = DataFrame({'Q_X2': ""},index=range(1, 18*periods + 1))
# df_Q_X3 = DataFrame({'Q_X3': ""},index=range(1, 18*periods + 1))
# df_Q_X4 = DataFrame({'Q_X4': ""},index=range(1, 18*periods + 1))
# df_Q_X5 = DataFrame({'Q_X5': ""},index=range(1, 18*periods + 1))
# df_Q_X6 = DataFrame({'Q_X6': ""},index=range(1, 18*periods + 1))
# df_Q_X7 = DataFrame({'Q_X7': ""},index=range(1, 18*periods + 1))

Q_aug_CCGT = np.zeros((1728,8))

# df_step_cost = DataFrame({'step_cost':""},index=range(1, periods + 1))
step_cost_result = np.zeros(96)

# df_PLF_r = DataFrame({'r':""},index=range(1,periods+1))             # save each segment r[a]
r_segment = np.zeros((96,5))

# df_a = DataFrame({'a':""},index=range(1,periods+1))                 # save each step's updated segment location
a_loc = np.zeros(96)

Partials = np.zeros(96)

# demand_Elec = demand_Elec_original
# demand_Heat = demand_Heat_original
# price = price_original
# P_WT_a = P_WT_a_original

for epoch in range(1,200):
    # # Step 3: Model Setup
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env()
    env.start()

    strategy = gp.Model("Deterministic SOC adp")

    from MC import Guass_Generation, Weibull_Generation
    # mean=0, variance=0.1
    demand_Elec = Guass_Generation(demand_Elec_original,0,0.1)
    demand_Heat = Guass_Generation(demand_Heat_original, 0, 0.1)
    # mean=0, variance=0.00001
    price = Guass_Generation(price_original,0,0.00001)
    # shape=1, scale=0.1
    P_WT_a = Weibull_Generation(P_WT_a_original,1,0.1)


    # Variable Definition

    P_FC = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.8, ub=7, name="P_FC")

    g_CCGT = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0.9918, ub=3.306, name="g_CCGT")

    P_GRID = strategy.addVar(vtype=GRB.CONTINUOUS, lb=-6, ub=6, name="P_GRID")

    P_c = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name="P_c")

    u_c = strategy.addVar(vtype=GRB.BINARY, name="u_c")

    P_d = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name="P_d")

    u_d = strategy.addVar(vtype=GRB.BINARY, name="u_d")

    P_wcur = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_wcur")

    P_cur = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_cur")

    Q_GB = strategy.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=15, name="Q_GB")

    Q_HP = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5, name="Q_HP")

    Q_cur = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Q_cur")

    SOC = strategy.addVar(vtype=GRB.CONTINUOUS, lb=1.5, ub=15, name="SOC")

    Q_waste = strategy.addVar(vtype=GRB.CONTINUOUS, name="Q_waste")


    Intervals_CCGT = range(1, 19)
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

    r = {}
    ub = (15-1.5)/Nt
    count_r = range(1,Nt+1)
    for a in count_r:
        r[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=ub, name='r.{}'.format(a))

    strategy.update()


    ## Start interval = 1
    interval = 1
    # Demand Constraints
    # 1) Q_CCGT
    for Interval in Intervals_CCGT:
        if Interval >= 2:
            strategy.addConstr(Q_X1[Interval] - Q_X2[Interval - 1] == 0)
            strategy.addConstr(Q_X2[Interval] - Q_X3[Interval - 1] == 0)
            strategy.addConstr(Q_X3[Interval] - Q_X4[Interval - 1] == 0)
            strategy.addConstr(Q_X4[Interval] - Q_X5[Interval - 1] == 0)
            strategy.addConstr(Q_X5[Interval] - Q_X6[Interval - 1] == 0)
            strategy.addConstr(Q_X6[Interval] - Q_X7[Interval - 1] == 0)
            strategy.addConstr(Q_X7[Interval] - 0.257*Q_X4[Interval - 1] + 0.3266*Q_X5[Interval - 1] \
                               + 0.6292*Q_X6[Interval - 1] -1.63*Q_X7[Interval-1] - g_CCGT == 0)

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

    # 2) Equations
    # print(demand_Elec[interval])
    strategy.addConstr((P_FC+ (g_CCGT * 16 - 10) + P_GRID + P_d + P_cur + (-P_wcur) - P_c\
                       -Q_HP / 4.5) + P_WT_a[interval-1] - demand_Elec[interval-1] == 0)
    strategy.addConstr((Q_GB + Q_HP + Q_cur) + (0.4031*Q_X1[18] + 0.3656*Q_X2[18] + 0.06311*Q_X3[18] + 0.2087*Q_X4[18])\
                       +(-Q_waste)- demand_Heat[interval-1] == 0)
    strategy.update()

    # Battery Constraints
    strategy.addConstr(SOC - 7.5 - 0.25 * (P_c * 0.9 - P_d / 0.9) == 0)
    strategy.addConstr(u_c + u_d <= 1)
    strategy.addConstr(P_c - u_c * 3 <= 0)
    strategy.addConstr(P_d - u_d * 3 <= 0)

    strategy.update()

    #  Curtailment Constraints
    strategy.addConstr(P_wcur - P_WT_a[interval-1] <= 0)
    strategy.addConstr(P_cur - demand_Elec[interval-1] <= 0)
    strategy.addConstr(Q_cur - demand_Heat[interval-1] <= 0)
    strategy.addConstr(Q_waste - demand_Heat[interval - 1] <= 0)

    strategy.update()

    # r[a] Constraints
    rr = np.array(Series(r))
    strategy.addConstr(sum(rr,1.5) == SOC)
    strategy.update()

    # Objective
    approximate = sum(slopes[interval-1]*rr)

    step_cost = 0.25 * ((types[0] * P_FC + types[1] * Q_GB + types[2] * g_CCGT) \
                        + (types[3] * P_wcur + types[4] * P_cur + types[5] * Q_cur) \
                        + types[6] * Q_waste + types[7] * SOC \
                        + 1000 * price[interval-1] * P_GRID) + approximate

    strategy.update()
    strategy.setParam('OutputFlag', 0)

    strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
    strategy.update()

    strategy.optimize()

    # Save step cost

    step_cost_result[interval-1]=strategy.ObjVal
###########################################################################
    temp_sol=[]
    for v in strategy.getVars():
        temp_sol.append(v.x)
    decisions[interval-1,0:13] = temp_sol[0:13]

    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 0] = strategy.getAttr('x', Q_X1).select()
    # print(Q_aug_CCGT[:,0])
    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 1] = strategy.getAttr('x', Q_X2).select()
    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 2] = strategy.getAttr('x', Q_X3).select()
    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 3] = strategy.getAttr('x', Q_X4).select()
    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 4] = strategy.getAttr('x', Q_X5).select()
    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 5] = strategy.getAttr('x', Q_X6).select()
    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 6] = strategy.getAttr('x', Q_X7).select()
    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 7] = 0.4031*Q_aug_CCGT[(interval - 1) * 18 : interval*18,0]\
                                                       + 0.3656*Q_aug_CCGT[(interval - 1) * 18 : interval*18,1] \
                                                       + 0.06311*Q_aug_CCGT[(interval - 1) * 18 : interval*18,2] \
                                                       + 0.2087*Q_aug_CCGT[(interval - 1) * 18 : interval*18,3]

    decisions[interval-1,13] = 0.25*1000 * price[interval-1] * P_GRID.x
    decisions[interval-1,14] = g_CCGT.x * 16 - 10
    decisions[interval-1,15] = Q_aug_CCGT[(interval - 1) * 18 + 17, 7]
    decisions[interval - 1, 16] = Q_waste.x

    r_segment[interval-1,:] = strategy.getAttr('x',r).select()

    approximate_1 = sum(slopes[interval - 1] * r_segment[interval - 1])
    # print(approximate_1)
    step_cost_result[interval - 1] -= approximate_1
    # print(step_cost_result[interval - 1])

    # Locate which slope d_a should be updated
    SOC_update_a = (decisions[interval-1,12] - 1.5) // ub
    if SOC_update_a == 5:
        a_loc[interval-1] = 4
    elif SOC_update_a == -1:
        a_loc[interval-1] = 0
    else:
        a_loc[interval-1] = SOC_update_a


    for interval in range(2, periods+1):
        env = gp.Env()
        env.start()
        strategy = gp.Model("stochastic adp%g"%interval)

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

        SOC = strategy.addVar(vtype=GRB.CONTINUOUS,lb=1.5, ub=15, name="SOC")

        Q_waste = strategy.addVar(vtype=GRB.CONTINUOUS, name="Q_waste")

        Intervals_CCGT = range(1, 19)
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

        r = {}
        ub = (15 - 1.5) / Nt
        count_r = range(1, Nt + 1)
        for a in count_r:
            r[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub, name='r.{}'.format(a))

        strategy.update()

        # Demand Constraints
        # 1) Q_CCGT
        for Interval in Intervals_CCGT:
            if Interval == 1:
                strategy.addConstr(Q_X1[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 1] == 0,'new1')  #q_x1(k+1)=q_x2(k)
                strategy.addConstr(Q_X2[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 2] == 0,'new2')
                strategy.addConstr(Q_X3[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 3] == 0,'new3')
                strategy.addConstr(Q_X4[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 4] == 0,'new4')
                strategy.addConstr(Q_X5[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 5] == 0,'new5')
                strategy.addConstr(Q_X6[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 6] == 0,'new6')
                strategy.addConstr(Q_X7[Interval] - 0.257 * Q_aug_CCGT[(interval-1)*18 - 1,3] + 0.3266 * Q_aug_CCGT[(interval-1)*18 - 1,4] \
                                   + 0.6292 * Q_aug_CCGT[(interval-1)*18 - 1,5] - 1.63 * Q_aug_CCGT[(interval-1)*18 - 1,6] - decisions[interval-2,1] == 0,'new7') #g_CCGT
            elif Interval >= 2:
                strategy.addConstr(Q_X1[Interval] - Q_X2[Interval - 1] == 0)
                strategy.addConstr(Q_X2[Interval] - Q_X3[Interval - 1] == 0)
                strategy.addConstr(Q_X3[Interval] - Q_X4[Interval - 1] == 0)
                strategy.addConstr(Q_X4[Interval] - Q_X5[Interval - 1] == 0)
                strategy.addConstr(Q_X5[Interval] - Q_X6[Interval - 1] == 0)
                strategy.addConstr(Q_X6[Interval] - Q_X7[Interval - 1] == 0)
                strategy.addConstr(Q_X7[Interval] - 0.257 * Q_X4[Interval - 1] + 0.3266 * Q_X5[Interval - 1] \
                                   + 0.6292 * Q_X6[Interval - 1] - 1.63 * Q_X7[Interval-1] - g_CCGT == 0)

        strategy.update()

        for Interval in Intervals_CCGT:
            strategy.addConstr(0.4031 * Q_X1[Interval] + 0.3656 * Q_X2[Interval] + 0.06311 * Q_X3[Interval] + 0.2087 * Q_X4[Interval] <= 50)
            strategy.addConstr(0.4031 * Q_X1[Interval] + 0.3656 * Q_X2[Interval] + 0.06311 * Q_X3[Interval] + 0.2087 * Q_X4[Interval] >= 15)

            if Interval >= 2:
                strategy.addConstr(0.4031 * (Q_X1[Interval] - Q_X1[Interval - 1]) + 0.3656 * (Q_X2[Interval] - Q_X2[Interval - 1]) \
                                   + 0.06311 * (Q_X3[Interval] - Q_X3[Interval - 1]) + 0.2087 * (Q_X4[Interval] - Q_X4[Interval - 1]) <= 1.5)
                strategy.addConstr(0.4031 * (Q_X1[Interval] - Q_X1[Interval - 1]) + 0.3656 * (Q_X2[Interval] - Q_X2[Interval - 1]) \
                                   + 0.06311 * (Q_X3[Interval] - Q_X3[Interval - 1]) + 0.2087 * (Q_X4[Interval] - Q_X4[Interval - 1]) >= -1.5)
        strategy.update()

        # 2) Equations
        strategy.addConstr((P_FC + (g_CCGT * 16 - 10) + P_GRID + P_d + P_cur + (-P_wcur) - P_c \
                            - Q_HP / 4.5) + P_WT_a[interval-1] - demand_Elec[interval-1] == 0)
        strategy.addConstr((Q_GB + Q_HP + Q_cur) + (0.4031*Q_X1[18] + 0.3656*Q_X2[18]+ 0.06311*Q_X3[18] + 0.2087*Q_X4[18]) \
                           +(-Q_waste)- demand_Heat[interval-1] == 0)

        strategy.update()
        # Ramping Constraints
        strategy.addConstr(g_CCGT - decisions[interval-2,1] <= 1.5)
        strategy.addConstr(g_CCGT - decisions[interval-2,1] >= -1.5)

        strategy.update()
        # Battery Constraints
        strategy.addConstr(SOC - decisions[interval-2,12] - 0.25 * (P_c * 0.9 - P_d / 0.9) == 0,'battery')
        strategy.addConstr(u_c + u_d <= 1)
        strategy.addConstr(P_c - u_c * 3 <= 0,'battery1')
        strategy.addConstr(P_d - u_d * 3 <= 0,'battery2')

        strategy.update()
        #  Curtailment Constraints
        strategy.addConstr(P_wcur - P_WT_a[interval-1] <= 0)
        strategy.addConstr(P_cur - demand_Elec[interval-1] <= 0)
        strategy.addConstr(Q_cur - demand_Heat[interval-1] <= 0)
        strategy.addConstr(Q_waste - demand_Heat[interval - 1] <= 0)

        strategy.update()

        # r[a] Constraints
        rr = np.array(Series(r))
        strategy.addConstr(sum(rr,1.5) == SOC)

        strategy.update()

        # Objective
        approximate = sum(slopes[interval-1] * rr)
        step_cost = 0.25 * ((types[0] * P_FC + types[1] * Q_GB + types[2] * g_CCGT) \
                        + (types[3] * P_wcur + types[4] * P_cur + types[5] * Q_cur) \
                        + types[6] * Q_waste + types[7] * SOC \
                        + 1000 * price[interval-1] * P_GRID) + approximate

        strategy.update()

        strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
        strategy.update()

        strategy.setParam('OutputFlag', 0)
        strategy.optimize()
        S1 = strategy.ObjVal

        # Save results
        step_cost_result[interval-1] = strategy.ObjVal

        temp_sol=[]
        for v in strategy.getVars():
            temp_sol.append(v.x)
        decisions[interval - 1, 0:13] = temp_sol[0:13]

        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 0] = strategy.getAttr('x', Q_X1).select()
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 1] = strategy.getAttr('x', Q_X2).select()
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 2] = strategy.getAttr('x', Q_X3).select()
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 3] = strategy.getAttr('x', Q_X4).select()
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 4] = strategy.getAttr('x', Q_X5).select()
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 5] = strategy.getAttr('x', Q_X6).select()
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 6] = strategy.getAttr('x', Q_X7).select()
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 7] = 0.4031 * Q_aug_CCGT[(interval - 1) * 18: interval * 18,0] \
                                                            + 0.3656 * Q_aug_CCGT[(interval - 1) * 18: interval * 18, 1] \
                                                            + 0.06311 * Q_aug_CCGT[(interval - 1) * 18: interval * 18, 2] \
                                                            + 0.2087 * Q_aug_CCGT[(interval - 1) * 18: interval * 18, 3]

        decisions[interval - 1, 13] = 1000 * price[interval - 1] * P_GRID.x
        decisions[interval - 1, 14] = g_CCGT.x * 16 - 10
        decisions[interval - 1, 15] = Q_aug_CCGT[(interval - 1) * 18 + 17, 7]
        decisions[interval - 1, 16] = Q_waste.x

        r_segment[interval - 1, :] = strategy.getAttr('x', r).select()

        approximate_1 = sum(slopes[interval-1] * r_segment[interval - 1])
        step_cost_result[interval - 1] -= approximate_1

        # Locate which slope d_a should be updated
        SOC_update_a = (decisions[interval - 1, 12] - 1.5) // ub
        if SOC_update_a == 5:
            a_loc[interval - 1] = 4
        elif SOC_update_a == -1:
            a_loc[interval - 1] = 0
        else:
            a_loc[interval - 1] = SOC_update_a

        #  Obtain an observation of the marginal value(shadow price) by solve the numerical derivative
        change = strategy
        battery = change.getConstrByName('battery')

        if decisions[interval-1,12] >= 14:
            battery.setAttr('rhs',(-1+decisions[interval-2,12]))
            change.update()
            change.optimize()

            S2 = change.ObjVal
            Partials[interval-1] = S1-S2
        else:
            battery.setAttr('rhs', (1 + decisions[interval - 2, 12]))
            change.update()
            change.optimize()

            S3 = change.ObjVal
            Partials[interval - 1] = (S3 - S1)




    # print(Partials)

    Cost.append(step_cost_result.sum())

    # 跨文件调用斜率更新py文件，将待引用的py文件路径加到了搜索列表里

    ### Backward Update slopes
    from spar import spar_update

    b = 5      #tunable parameters, b越大，前期越相信更新的值（越不相信上一轮更新的结果）
    step_size = b/(b+epoch-1)       # Generalized Harmonic Stepsizes

    for interval in range(1,periods):
        a = a_loc[interval-1]
        # updated_slope = step_size * Partials[interval]+ (1-step_size) * slopes[interval-1, int(a)]
        updated_slope = step_size * Partials[interval:interval + 5].min() + (1 - step_size) * slopes[interval - 1, int(a)]

        # discount_factor = 0.98
        # discount_index = np.argmin(Partials[interval:])
        # updated_slope = step_size * Partials[interval:].min() * np.power(discount_factor, discount_index)\
        #                 + (1 - step_size) * slopes[interval - 1, int(a)]
        # updated_slope = step_size * Partials[interval:].mean() + (1 - step_size) * slopes[interval - 1, int(a)]

        # print(updated_slope)

        # update the new slope into df_slopes
        slopes[interval-1, int(a)] = updated_slope
        located_bars = slopes[interval-1,:]
        # print(located_bars)
        slopes[interval-1,:] = np.array(spar_update(located_bars, int(a)))


print(np.mean(Cost[epoch-50:epoch]))
# print(r_segment)
# print(decisions[:,12])
# print(slopes)
# print(Partials)


# outputpath = 'F:\CCGT\ADP\\1225ADP_result.csv'
# df_decision_vars.to_csv(outputpath, sep=',', index=True, header=True)
#
# outputpath1 = 'F:\CCGT\ADP\\1225ADP_Q_CCGT_all.csv'
# df_Q_CCGT.to_csv(outputpath1, sep=',', index=True, header=True)
#
# outputpath2 = 'F:\CCGT\ADP\\1225ADP_GRID_PRICE.csv'
# df_GRID_PRICE.to_csv(outputpath2, sep=',', index=True, header=True)
#
# outputpath3 = 'F:\CCGT\ADP\\1225ADP_step_cost.csv'
# df_step_cost.to_csv(outputpath3, sep=',', index=True, header=True)
#
# outputpath4 = 'F:\CCGT\ADP\\1225ADP_P_CCGT.csv'
# df_P_CCGT.to_csv(outputpath4, sep=',', index=True, header=True)
#
# outputpath5 = 'F:\CCGT\ADP\\1225ADP_slopes.csv'
# df_slopes.to_csv(outputpath5, sep=',', index=True, header=True)


# ###------Stacked bar chart 堆叠柱形图-------###
# 1) Power
P_GRID = decisions[:,2]
P_CCGT = decisions[:,14]
P_FC = decisions[:,0]
SOC = decisions[:,12]
P_wcur = decisions[:,7]
P_cur = decisions[:,8]

P_HP = decisions[:,10]/4.5
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
ax1.set_xticklabels(intervals)
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
ax1.set_title('ADP Power Assignment with SOC',fontsize = 25, fontname='Times New Roman', y = 1.05)

# 2) Heat
fig, ax = plt.subplots()


Q_GB = decisions[:,9]
Q_HP = decisions[:,10]
Q_CCGT_sol = decisions[:,15]
Q_cur = decisions[:,11]
Q_waste = decisions[:,16]

bar1 = ax.bar(intervals, Q_HP, width, label='Q_HP',color='gold')
bar2 = ax.bar(intervals, Q_GB, width, bottom = Q_HP,label='Q_GB',color='deepskyblue')
bar3 = ax.bar(intervals, Q_CCGT_sol, width, bottom = (Q_GB+Q_HP), label='Q_CCGT',color='salmon')
bar4 = ax.bar(intervals, Q_cur, width,bottom = (Q_GB+Q_HP+Q_CCGT_sol), label='Q_cur',facecolor='white',edgecolor='black',hatch='\\\\')
bar5 = ax.bar(intervals, -Q_waste, width,bottom = (Q_GB+Q_HP+Q_CCGT_sol+Q_cur), label='Q_waste',facecolor='white',edgecolor='black',hatch='//')
str = ax.plot(intervals, demand_Heat, label='Heat_Demand',color='b',linewidth=2)

# 设置x轴刻度
ax.set_xticks(intervals)
ax.set_xticklabels(intervals)
ax.set_xticklabels(intervals, rotation=50)
ax.set_xlim(-0.5,97)
# 设置x，y轴
ax.set_ylabel("Q(MW)", fontsize = 20, fontname='Times New Roman')
ax.set_xlabel("Time Periods (interval=15min)", fontsize = 20, fontname='Times New Roman')
# 显式图例
ax.legend()
# 显示标题
ax.set_title('ADP Heat Assignment with SOC', fontsize = 25, fontname='Times New Roman', y = 1.05)
# 显示网格
fig.tight_layout()
plt.grid(axis="y", linestyle='-.')

# 3) Q_CCGT
fig, ax = plt.subplots()
ax.plot(range(1, 18 * periods + 1),Q_aug_CCGT[:,7], linewidth=5)

#设置x，y轴
ax.set_ylabel("Q_CCGT(MW)", fontsize = 20,fontname='Times New Roman')
ax.set_xlabel("Time Periods (interval=50s)", fontsize = 20,fontname='Times New Roman')
ax.set_xlim(-20,1780)
# # 显式图例
# ax.legend()
# 显示网格
plt.grid(linestyle='-.')
fig.tight_layout()
# # 显示标题
# ax.set_title('Q_CCGT_all',fontsize = 15, y = 1.05)

#4) Cost
fig, ax = plt.subplots()
ax.plot(Cost)

plt.show()
# print(df_slopes)

#论文画图见plot文件


