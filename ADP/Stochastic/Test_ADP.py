import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

# adp_cost = np.load("D:\\Nuts\CCGT\ADP\Stochastic\\adp_cost.npy")
# milp_cost = np.load("D:\\Nuts\CCGT\ADP\Stochastic\\milp_cost.npy")
# error = list(map(lambda x:(x[1]-x[0])/x[0],zip(milp_cost,adp_cost)))

# np.mean(error[:200])
# 0.009528692410147397
# np.mean(error[200:400])
# 0.00480975333437964
# np.mean(error[400:600])
# 0.004629984767064912
# np.mean(error[600:800])
# 0.003793119310104171



# np.set_printoptions(threshold=np.inf)

rcParams['figure.figsize'] = 11, 5
pd.options.display.max_rows = None
pd.options.display.max_columns = None

Nt = 5      #记录分段数Nt
slopes_CCGT = np.load("D:\\Nuts\CCGT\ADP\Stochastic\\slopes_CCGT.npy")
slopes_SOC = np.load("D:\\Nuts\CCGT\ADP\Stochastic\\slopes_SOC.npy")

# # Step 2: Prepare the Data

# Prepare parameters

# types = ("P_FC", "Q_GB", "g_CCGT", "P_wcur", "P_cur", "Q_cur", "Q_waste", "SOC")
types = np.array((100, 30, 200, 200, 350, 350, 200, 0))

# # Prepare Demand Data
# df_DE = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
# df_DQ = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")
# periods = len(df_DE)
# print("periods = {}".format(periods))
# demand_Elec = np.array(df_DE['Edemand(MW)'])#E_plus
# demand_Heat = np.array(df_DQ['Qdemand(MW)'])##Q_plus_1
#
#
# # Prepare Exogenous Data
# df_P_WT = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\wind_turbine.xls")
# df_price = pd.read_excel("C:\\Users\\之之\\Desktop\\ADP_DATA\\forcasted_elec_price.xls")
# # df_P_WT = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\wind_turbine.xls")
# # df_price = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_elec_price.xls")
# P_WT_a = np.array(df_P_WT['Theoretical_Power_Curve (MW)'])
# price = np.array(df_price['price/$'])

periods = 96

# Create DataFrame
decisions = np.zeros((96,17))

Q_aug_CCGT = np.zeros((1728,8))
Q_temp = np.zeros((1728,8))

# df_step_cost = DataFrame({'step_cost':""},index=range(1, periods + 1))
step_cost_result = np.zeros(96)

# df_PLF_r = DataFrame({'r':""},index=range(1,periods+1))             # save each segment r[a]
r_segment_CCGT = np.zeros((96,5))
r_segment_SOC = np.zeros((96,5))

test_Cost = []

scenarios = np.load("D:\\Nuts\CCGT\ADP\Stochastic\\robustness_test.npy")
for i in range(1,31):
    demand_Elec = scenarios[i - 1, 0, :]
    demand_Heat = scenarios[i - 1, 1, :]
    # mean=0, variance=0.00001
    price = scenarios[i - 1, 2, :]
    # shape=1, scale=0.1
    P_WT_a = scenarios[i - 1, 3, :]

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env()
    env.start()

    strategy = gp.Model("stochastic adp")

    # Variable Definition

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

    # Nt = 5
    ##更改的分段
    count_r = range(1, Nt + 1)

    r_CCGT = {}
    ub_CCGT = (50-15)/Nt
    for a in count_r:
        r_CCGT[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub_CCGT, name='r.{}'.format(a))

    r_SOC = {}
    ub_SOC = (15-1.5)/Nt
    for a in count_r:
        r_SOC[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=ub_SOC, name='r.{}'.format(a))

    strategy.update()


    ## Start interval = 1
    interval = 1
    # Demand Constraints
    # 1) Q_CCGT
    for Interval in Intervals_CCGT:
        if Interval >=2:
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

        # if Interval == 1:
        #     strategy.addConstr(0.4031 * Q_X1[Interval] + 0.3656 * Q_X2[Interval] + 0.06311 * Q_X3[Interval] + 0.2087 * Q_X4[Interval]<=37)
        # else:
        if Interval >=2:
            strategy.addConstr(0.4031*(Q_X1[Interval]-Q_X1[Interval-1]) + 0.3656*(Q_X2[Interval]-Q_X2[Interval-1]) \
                                   + 0.06311*(Q_X3[Interval]-Q_X3[Interval-1]) + 0.2087*(Q_X4[Interval]-Q_X4[Interval-1]) <= 1.5)
            strategy.addConstr(0.4031*(Q_X1[Interval]-Q_X1[Interval-1]) + 0.3656*(Q_X2[Interval]-Q_X2[Interval-1]) \
                                   + 0.06311*(Q_X3[Interval]-Q_X3[Interval-1]) + 0.2087*(Q_X4[Interval]-Q_X4[Interval-1]) >= -1.5)
    strategy.update()

    # 2) Equations
    strategy.addConstr((P_FC+ (g_CCGT * 16 - 10) + P_GRID + P_d + P_cur + (-P_wcur) - P_c -Q_HP / 4.5) \
                       + P_WT_a[interval-1] - demand_Elec[interval-1] == 0)
    strategy.addConstr((Q_GB + Q_HP + Q_cur) + (0.4031*Q_X1[18] + 0.3656*Q_X2[18] + 0.06311*Q_X3[18] + 0.2087*Q_X4[18]) +(-Q_waste)\
                       - demand_Heat[interval-1] == 0)
    strategy.update()

    # Battery Constraints
    strategy.addConstr(SOC - 7.5 - 0.25 * (P_c * 0.9 - P_d / 0.9) == 0)

    strategy.addConstr(u_c + u_d <= 1)
    strategy.addConstr(P_c - u_c * 3 <= 0)
    strategy.addConstr(P_d - u_d * 3 <= 0)

    strategy.addConstr(SOC >=1.5)
    strategy.addConstr(SOC <=15)

    strategy.update()

    #  Curtailment Constraints
    strategy.addConstr(P_wcur - P_WT_a[interval-1] <= 0)
    strategy.addConstr(P_cur - demand_Elec[interval-1] <= 0)
    strategy.addConstr(Q_cur - demand_Heat[interval-1] <= 0)
    strategy.addConstr(Q_waste - demand_Heat[interval-1] <= 0)

    strategy.update()

    # r[a] Constraints
    ##更改
    rr_CCGT = np.array(Series(r_CCGT))
    strategy.addConstr(sum(rr_CCGT,15)==0.4031*Q_X1[18] + 0.3656*Q_X2[18] + 0.06311*Q_X3[18] + 0.2087*Q_X4[18])

    rr_SOC = np.array(Series(r_SOC))
    strategy.addConstr(sum(rr_SOC,1.5)==SOC)

    strategy.update()

    # Objective
    approximate_CCGT = sum(slopes_CCGT[interval-1]*rr_CCGT)
    approximate_SOC = sum(slopes_SOC[interval-1]*rr_SOC)

    step_cost = 0.25 * ((types[0] * P_FC + types[1] * Q_GB + types[2] * g_CCGT) \
                        + (types[3] * P_wcur + types[4] * P_cur + types[5] * Q_cur) \
                        + types[6] * Q_waste + types[7] * SOC \
                        + 1000 * price[interval-1] * P_GRID) + approximate_CCGT + approximate_SOC

    strategy.update()
    strategy.setParam('OutputFlag', 0)

    strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
    strategy.update()

    strategy.optimize()

    # Save step cost

    step_cost_result[interval-1]=strategy.ObjVal

    temp_sol=[]
    for v in strategy.getVars():
        temp_sol.append(v.x)
    decisions[interval-1,0:13] = temp_sol[0:13]

    Q_aug_CCGT[(interval - 1) * 18 : interval*18, 0] = strategy.getAttr('x', Q_X1).select()
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

    decisions[interval-1,13] = 1000 * price[interval-1] * P_GRID.x
    decisions[interval-1,14] = g_CCGT.x * 16 - 10
    decisions[interval-1,15] = Q_aug_CCGT[(interval - 1) * 18 + 17, 7]
    decisions[interval - 1, 16] = Q_waste.x

    r_segment_CCGT[interval-1,:] = strategy.getAttr('x',r_CCGT).select()
    r_segment_SOC[interval-1, :] = strategy.getAttr('x',r_SOC).select()

    approximate_1 = sum(slopes_CCGT[interval - 1] * r_segment_CCGT[interval - 1])+sum(slopes_SOC[interval-1]*r_segment_SOC[interval-1])
    # print(approximate_1)
    step_cost_result[interval - 1] -= approximate_1

    for interval in range(2, periods + 1):
        env = gp.Env()
        env.start()
        strategy = gp.Model("stochastic adp%g" % interval)

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

        SOC = strategy.addVar(vtype=GRB.CONTINUOUS, name="SOC")

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

        ##更改
        count_r = range(1, Nt + 1)

        r_CCGT = {}
        ub_CCGT = (50 - 15) / Nt
        for a in count_r:
            r_CCGT[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub_CCGT, name='r.{}'.format(a))

        r_SOC = {}
        ub_SOC = (15 - 1.5) / Nt
        for a in count_r:
            r_SOC[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub_SOC, name='r.{}'.format(a))

        strategy.update()

        # Demand Constraints
        # 1) Q_CCGT
        for Interval in Intervals_CCGT:
            if Interval == 1:
                strategy.addConstr(Q_X1[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 1] == 0,
                                   'new1')  # q_x1(k+1)=q_x2(k)
                strategy.addConstr(Q_X2[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 2] == 0, 'new2')
                strategy.addConstr(Q_X3[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 3] == 0, 'new3')
                strategy.addConstr(Q_X4[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 4] == 0, 'new4')
                strategy.addConstr(Q_X5[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 5] == 0, 'new5')
                strategy.addConstr(Q_X6[Interval] - Q_aug_CCGT[(interval - 1) * 18 - 1, 6] == 0, 'new6')
                strategy.addConstr(Q_X7[Interval] - 0.257 * Q_aug_CCGT[(interval - 1) * 18 - 1, 3] + 0.3266 * Q_aug_CCGT[
                    (interval - 1) * 18 - 1, 4] \
                                   + 0.6292 * Q_aug_CCGT[(interval - 1) * 18 - 1, 5] - 1.63 * Q_aug_CCGT[
                                       (interval - 1) * 18 - 1, 6] - decisions[interval - 2, 1] == 0, 'new7')  # g_CCGT
            elif Interval >= 2:
                strategy.addConstr(Q_X1[Interval] - Q_X2[Interval - 1] == 0)
                strategy.addConstr(Q_X2[Interval] - Q_X3[Interval - 1] == 0)
                strategy.addConstr(Q_X3[Interval] - Q_X4[Interval - 1] == 0)
                strategy.addConstr(Q_X4[Interval] - Q_X5[Interval - 1] == 0)
                strategy.addConstr(Q_X5[Interval] - Q_X6[Interval - 1] == 0)
                strategy.addConstr(Q_X6[Interval] - Q_X7[Interval - 1] == 0)
                strategy.addConstr(Q_X7[Interval] - 0.257 * Q_X4[Interval - 1] + 0.3266 * Q_X5[Interval - 1] \
                                   + 0.6292 * Q_X6[Interval - 1] - 1.63 * Q_X7[Interval - 1] - g_CCGT == 0)

        strategy.update()

        for Interval in Intervals_CCGT:
            strategy.addConstr(0.4031 * Q_X1[Interval] + 0.3656 * Q_X2[Interval] + 0.06311 * Q_X3[Interval] + 0.2087 * Q_X4[
                Interval] <= 50)
            strategy.addConstr(0.4031 * Q_X1[Interval] + 0.3656 * Q_X2[Interval] + 0.06311 * Q_X3[Interval] + 0.2087 * Q_X4[
                Interval] >= 15)

            if Interval >= 2:
                strategy.addConstr(
                    0.4031 * (Q_X1[Interval] - Q_X1[Interval - 1]) + 0.3656 * (Q_X2[Interval] - Q_X2[Interval - 1]) \
                    + 0.06311 * (Q_X3[Interval] - Q_X3[Interval - 1]) + 0.2087 * (
                                Q_X4[Interval] - Q_X4[Interval - 1]) <= 1.5)
                strategy.addConstr(
                    0.4031 * (Q_X1[Interval] - Q_X1[Interval - 1]) + 0.3656 * (Q_X2[Interval] - Q_X2[Interval - 1]) \
                    + 0.06311 * (Q_X3[Interval] - Q_X3[Interval - 1]) + 0.2087 * (
                                Q_X4[Interval] - Q_X4[Interval - 1]) >= -1.5)
        strategy.update()

        # 2) Equations
        strategy.addConstr((P_FC + (g_CCGT * 16 - 10) + P_GRID + P_d + P_cur + (-P_wcur) - P_c \
                            - Q_HP / 4.5) + P_WT_a[interval - 1] - demand_Elec[interval - 1] == 0)
        strategy.addConstr(
            (Q_GB + Q_HP + Q_cur) + (0.4031 * Q_X1[18] + 0.3656 * Q_X2[18] + 0.06311 * Q_X3[18] + 0.2087 * Q_X4[18]) \
            + (-Q_waste) - demand_Heat[interval - 1] == 0)

        strategy.update()
        # Ramping Constraints
        strategy.addConstr(g_CCGT - decisions[interval - 2, 1] <= 1.5)
        strategy.addConstr(g_CCGT - decisions[interval - 2, 1] >= -1.5)

        strategy.update()
        # Battery Constraints
        strategy.addConstr(SOC - decisions[interval - 2, 12] - 0.25 * (P_c * 0.9 - P_d / 0.9) == 0, 'battery')
        strategy.addConstr(u_c + u_d <= 1)
        strategy.addConstr(P_c - u_c * 3 <= 0)
        strategy.addConstr(P_d - u_d * 3 <= 0)

        strategy.update()

        strategy.addConstr(SOC <= 15, 'SOC_ub')
        strategy.addConstr(SOC >= 1.5)

        strategy.update()
        #  Curtailment Constraints
        strategy.addConstr(P_wcur - P_WT_a[interval - 1] <= 0)
        strategy.addConstr(P_cur - demand_Elec[interval - 1] <= 0)
        strategy.addConstr(Q_cur - demand_Heat[interval - 1] <= 0)
        strategy.addConstr(Q_waste - demand_Heat[interval - 1] <= 0)

        strategy.update()

        # r[a] Constraints
        ##更改
        rr_CCGT = np.array(Series(r_CCGT))
        strategy.addConstr(
            sum(rr_CCGT, 15) == 0.4031 * Q_X1[18] + 0.3656 * Q_X2[18] + 0.06311 * Q_X3[18] + 0.2087 * Q_X4[18])

        rr_SOC = np.array(Series(r_SOC))
        strategy.addConstr(sum(rr_SOC, 1.5) == SOC)

        strategy.update()

        # Objective
        approximate_CCGT = sum(slopes_CCGT[interval - 1] * rr_CCGT)
        approximate_SOC = sum(slopes_SOC[interval - 1] * rr_SOC)

        step_cost = 0.25 * ((types[0] * P_FC + types[1] * Q_GB + types[2] * g_CCGT) \
                            + (types[3] * P_wcur + types[4] * P_cur + types[5] * Q_cur) \
                            + types[6] * Q_waste + types[7] * SOC \
                            + 1000 * price[interval - 1] * P_GRID) + approximate_CCGT + approximate_SOC
        strategy.update()

        strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
        strategy.update()

        strategy.setParam('OutputFlag', 0)
        strategy.optimize()
        S1 = strategy.ObjVal

        # Save results
        step_cost_result[interval - 1] = strategy.ObjVal
        # print(step_cost_result[interval-1])

        temp_sol = []
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
        Q_aug_CCGT[(interval - 1) * 18: interval * 18, 7] = 0.4031 * Q_aug_CCGT[(interval - 1) * 18: interval * 18, 0] \
                                                            + 0.3656 * Q_aug_CCGT[(interval - 1) * 18: interval * 18, 1] \
                                                            + 0.06311 * Q_aug_CCGT[(interval - 1) * 18: interval * 18, 2] \
                                                            + 0.2087 * Q_aug_CCGT[(interval - 1) * 18: interval * 18, 3]

        decisions[interval - 1, 13] = 1000 * price[interval - 1] * P_GRID.x
        decisions[interval - 1, 14] = g_CCGT.x * 16 - 10
        decisions[interval - 1, 15] = Q_aug_CCGT[(interval - 1) * 18 + 17, 7]
        decisions[interval - 1, 16] = Q_waste.x

        r_segment_CCGT[interval - 1, :] = strategy.getAttr('x', r_CCGT).select()
        r_segment_SOC[interval - 1, :] = strategy.getAttr('x', r_SOC).select()

        approximate_1 = sum(slopes_CCGT[interval - 1] * r_segment_CCGT[interval - 1]) + sum(
            slopes_SOC[interval - 1] * r_segment_SOC[interval - 1])
        # print(approximate_1)
        step_cost_result[interval - 1] -= approximate_1

    test_Cost.append(step_cost_result.sum())

print(test_Cost)
np.save('D:\\Nuts\CCGT\ADP\Stochastic\\adp_robust_cost_e3.npy', test_Cost)