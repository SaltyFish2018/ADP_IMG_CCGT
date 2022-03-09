

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pylab import rcParams


rcParams['figure.figsize'] = 11, 5
pd.options.display.max_rows = None
pd.options.display.max_columns = None

df_slopes = DataFrame({'1':"",'2':"",'3':""},index=range(1, 96 + 1))         #slopes总表，保存第n次迭代后的更新结果
df_slopes.loc[:,:]=0
df_slopes.loc[:,:]=df_slopes.loc[:,:].astype(float)

Cost = []    #第n次迭代中的目标函数

# # Step 2: Prepare the Data

# Prepare parameters

types = ("P_FC", "Q_GB", "g_CCGT", "P_wcur", "P_cur", "Q_cur")

adjustable_parameters_data = {"param1": [65, 300, 1460 / 3600 * 3600, 200, 150, 350]}

df_types = DataFrame(adjustable_parameters_data, index=types)


# Prepare Demand Data
df_DE = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_electricity_demand.xls")
df_DQ = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_heat_demand.xls")
periods = len(df_DE)
print("periods = {}".format(periods))
demand_Elec = Series(df_DE['E_plus'])
# demand_Elec = Series(df_DE['Edemand(MW)'])
# demand_Heat = Series(df_DQ['Qdemand(MW)'])
demand_Heat = Series(df_DQ['Q_plus'])
demand_Elec.index = range(1, periods + 1)
demand_Heat.index = range(1, periods + 1)

# Prepare Exogenous Data
df_P_WT = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\wind_turbine.xls")
P_WT_a = Series(df_P_WT['Theoretical_Power_Curve (MW)'])
P_WT_a.index = range(1, periods + 1)

df_price = pd.read_excel("C:\\Users\\69560\\Desktop\\ADP_DATA\\forcasted_elec_price.xls")
price = Series(df_price['price/$'])
price.index = range(1, periods + 1)

# Create DataFrame
df_decision_vars = DataFrame({'P_FC': "", 'g_CCGT': "", 'P_GRID':"", 'P_c': "", 'u_c': "", 'P_d': "", 'u_d': "", 'P_wcur':"",
                              'P_cur': "", 'Q_GB': "", 'Q_HP': "", 'Q_cur': "", 'SOC': ""},index=range(1, periods + 1))

df_GRID_PRICE = DataFrame({'GRID_PRICE': ""},index=range(1, periods + 1))
df_P_CCGT = DataFrame({'P_CCGT': ""},index=range(1, periods + 1))

df_Q_CCGT = DataFrame({'Q_CCGT': ""},index=range(1, 18*periods + 1))
df_Q_X1 = DataFrame({'Q_X1': ""},index=range(1, 18*periods + 1))
df_Q_X2 = DataFrame({'Q_X2': ""},index=range(1, 18*periods + 1))
df_Q_X3 = DataFrame({'Q_X3': ""},index=range(1, 18*periods + 1))
df_Q_X4 = DataFrame({'Q_X4': ""},index=range(1, 18*periods + 1))
df_Q_X5 = DataFrame({'Q_X5': ""},index=range(1, 18*periods + 1))
df_Q_X6 = DataFrame({'Q_X6': ""},index=range(1, 18*periods + 1))
df_Q_X7 = DataFrame({'Q_X7': ""},index=range(1, 18*periods + 1))

df_step_cost = DataFrame({'step_cost':""},index=range(1, periods + 1))

df_PLF_r = DataFrame({'r':""},index=range(1,periods+1))             # save each segment r[a]
df_a = DataFrame({'a':""},index=range(1,periods+1))                 # save each step's updated segment location
df_shadow_price = DataFrame({'d':""},index=range(1,periods+1))      # save each step's shadow price for Q_CCGT(1)

# Set index names
df_decision_vars.index.names = ['intervals']
df_GRID_PRICE.index.names = ['intervals']
df_P_CCGT.index.names = ['intervals']
df_Q_CCGT.index.names = ['Intervals']
df_step_cost.index.names = ['intervals']
df_PLF_r.index.names = ['intervals']
df_a.index.names = ['intervals']
df_shadow_price.index.names = ['intervals']


for n in range(1,50):
    # # Step 3: Model Setup

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env()
    env.start()

    strategy = gp.Model("deterministic adp")

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
    Nt = 3
    ub = (50-15)/Nt
    count_r = range(1,Nt+1)
    for a in count_r:
        r[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=ub, name='r.{}'.format(a))

    Q = strategy.addVar(vtype=GRB.CONTINUOUS, lb=15, ub=50, name="Q")

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
                       -Q_HP / 4.5) + P_WT_a[interval] - demand_Elec[interval] == 0)
    strategy.addConstr((Q_GB + Q_HP + Q_cur) + (0.4031*Q_X1[18] + 0.3656*Q_X2[18] + 0.06311*Q_X3[18] + 0.2087*Q_X4[18]) - demand_Heat[interval] == 0)
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
    strategy.addConstr(P_wcur - P_WT_a[interval] <= 0)
    strategy.addConstr(P_cur - demand_Elec[interval] <= 0)
    strategy.addConstr(Q_cur - demand_Heat[interval] <= 0)

    strategy.update()

    # Q Constraint
    strategy.addConstr(0.4031*Q_X1[1] + 0.3656*Q_X2[1] + 0.06311*Q_X3[1] + 0.2087*Q_X4[1]==Q,'new')

    # r[a] Constraints
    rr = np.array(Series(r))
    strategy.addConstr(sum(rr,15)==0.4031*Q_X1[18] + 0.3656*Q_X2[18] + 0.06311*Q_X3[18] + 0.2087*Q_X4[18])
    strategy.update()

    # Objective
    approximate = sum(df_slopes.loc[interval]*rr)

    step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * P_FC \
                         + df_types.loc["Q_GB"].param1 * Q_GB \
                         + df_types.loc["g_CCGT"].param1 * g_CCGT) \
                        + (df_types.loc["P_wcur"].param1 * P_wcur \
                           + df_types.loc["P_cur"].param1 * P_cur \
                           + df_types.loc["Q_cur"].param1 * Q_cur) \
                        + 1000 * price[interval] * P_GRID) + approximate

    strategy.update()


    strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
    strategy.update()
    strategy.optimize()

    # Save step cost

    # for v in strategy.getVars():
    #     print(' %s %g' % (v.varName, v.x))
    # print ('Obj: %g' % strategy.objVal )

    df_step_cost.loc[interval].step_cost=strategy.ObjVal

    for v in strategy.getVars():
        df_decision_vars.loc[interval,'%s'%v.varName] = v.x


    Q_X1_sol = strategy.getAttr('x', Q_X1)
    Q_X2_sol = strategy.getAttr('x', Q_X2)
    Q_X3_sol = strategy.getAttr('x', Q_X3)
    Q_X4_sol = strategy.getAttr('x', Q_X4)
    Q_X5_sol = strategy.getAttr('x', Q_X5)
    Q_X6_sol = strategy.getAttr('x', Q_X6)
    Q_X7_sol = strategy.getAttr('x', Q_X7)

    for i in Intervals_CCGT:
        df_Q_X1.loc[(interval - 1) * 18 + i].Q_X1 = Q_X1_sol[i]
        df_Q_X2.loc[(interval - 1) * 18 + i].Q_X2 = Q_X2_sol[i]
        df_Q_X3.loc[(interval - 1) * 18 + i].Q_X3 = Q_X3_sol[i]
        df_Q_X4.loc[(interval - 1) * 18 + i].Q_X4 = Q_X4_sol[i]
        df_Q_X5.loc[(interval - 1) * 18 + i].Q_X5 = Q_X5_sol[i]
        df_Q_X6.loc[(interval - 1) * 18 + i].Q_X6 = Q_X6_sol[i]
        df_Q_X7.loc[(interval - 1) * 18 + i].Q_X7 = Q_X7_sol[i]
        df_Q_CCGT.loc[(interval - 1) * 18 + i].Q_CCGT = 0.4031*Q_X1_sol[i] + 0.3656*Q_X2_sol[i] + 0.06311*Q_X3_sol[i] + 0.2087*Q_X4_sol[i]

    df_P_CCGT.loc[interval].P_CCGT = g_CCGT.x * 16 - 10
    df_GRID_PRICE.loc[interval].GRID_PRICE = 1000 * price[interval] * P_GRID.x

    Q_sol = Q.X
    r_sol = strategy.getAttr('x',r)
    df_PLF_r.loc[interval].r = r_sol

    # Locate which slope d_a should be updated
    Q_update_a = (df_Q_CCGT.loc[interval*18].Q_CCGT-15)//ub + 1
    df_a.loc[interval].a = Q_update_a

    # #  Obtain an observation of the marginal value(shadow price) by solve the fixed dual problem
    # fixed = strategy.fixed()
    # Q_Var=fixed.getVarByName('Q')
    # Q_Var.setAttr('LB',0)
    # Q_Var.setAttr('UB',0)
    # fixed.update()
    #
    # cstr = fixed.getConstrByName('new')
    # cstr.setAttr('rhs',Q_sol)
    # fixed.update()
    #
    # fixed.optimize()
    # for v in fixed.getVars():
    #     print(' %s %g' % (v.varName, v.x))
    # print ('Obj: %g' % fixed.objVal )

    # shadowprice = cstr.getAttr("Pi")
    # df_shadow_price.loc[interval].d = shadowprice

    # shadowprice = fixed.getAttr("Pi",fixed.getConstrs())
    # keys = range(1,203)
    # value = zip(keys,shadowprice)
    # print(dict(value))

    df_Partial = DataFrame({'Partial': ""},index=range(1, periods + 1))
    df_Partial.index.names = ['intervals']
    df_Partial.loc[:,:] = 0
    df_Partial.loc[:,:]=df_Partial.loc[:,:].astype(float)

    for interval in range(2, periods+1):
    # for interval in range(2, 4):
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
        Nt = 3
        ub = (50 - 15) / Nt
        count_r = range(1, Nt + 1)
        for a in count_r:
            r[a] = strategy.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub, name='r.{}'.format(a))

        strategy.update()

        # Demand Constraints
        # 1) Q_CCGT
        for Interval in Intervals_CCGT:
            if Interval == 1:
                strategy.addConstr(Q_X1[Interval] - df_Q_X2.loc[(interval - 1) * 18].Q_X2 == 0,'new1')
                strategy.addConstr(Q_X2[Interval] - df_Q_X3.loc[(interval - 1) * 18].Q_X3 == 0,'new2')
                strategy.addConstr(Q_X3[Interval] - df_Q_X4.loc[(interval - 1) * 18].Q_X4 == 0,'new3')
                strategy.addConstr(Q_X4[Interval] - df_Q_X5.loc[(interval - 1) * 18].Q_X5 == 0,'new4')
                strategy.addConstr(Q_X5[Interval] - df_Q_X6.loc[(interval - 1) * 18].Q_X6 == 0,'new5')
                strategy.addConstr(Q_X6[Interval] - df_Q_X7.loc[(interval - 1) * 18].Q_X7 == 0,'new6')
                strategy.addConstr(Q_X7[Interval] - 0.257 * df_Q_X4.loc[(interval - 1) * 18].Q_X4 + 0.3266 * df_Q_X5.loc[(interval - 1) * 18].Q_X5 \
                                   + 0.6292 * df_Q_X6.loc[(interval - 1) * 18].Q_X6 - 1.63 * df_Q_X7.loc[(interval - 1) * 18].Q_X7 - df_decision_vars.loc[interval-1].g_CCGT == 0,'new7')
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
                            - Q_HP / 4.5) + P_WT_a[interval] - demand_Elec[interval] == 0)
        strategy.addConstr((Q_GB + Q_HP + Q_cur) + (0.4031*Q_X1[18] + 0.3656*Q_X2[18]+ 0.06311*Q_X3[18] + 0.2087*Q_X4[18]) - demand_Heat[interval] == 0)

        strategy.update()
        # Ramping Constraints
        strategy.addConstr(g_CCGT - df_decision_vars.loc[interval - 1].g_CCGT <= 1.5)
        strategy.addConstr(g_CCGT - df_decision_vars.loc[interval - 1].g_CCGT >= -1.5)

        strategy.update()
        # Battery Constraints
        strategy.addConstr(SOC - df_decision_vars.loc[interval - 1].SOC - 0.25 * (P_c * 0.9 - P_d / 0.9) == 0,'battery')
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

        # r[a] Constraints
        rr = np.array(Series(r))
        strategy.addConstr(sum(rr, 15) == 0.4031 * Q_X1[18] + 0.3656 * Q_X2[18] + 0.06311 * Q_X3[18] + 0.2087 * Q_X4[18])

        strategy.update()

        # Objective
        approximate = sum(df_slopes.loc[interval] * rr)
        step_cost = 0.25 * ((df_types.loc["P_FC"].param1 * P_FC \
                             + df_types.loc["Q_GB"].param1 * Q_GB \
                             + df_types.loc["g_CCGT"].param1 * g_CCGT) \
                            + (df_types.loc["P_wcur"].param1 * P_wcur \
                               + df_types.loc["P_cur"].param1 * P_cur \
                               + df_types.loc["Q_cur"].param1 * Q_cur) \
                            + 1000 * price[interval] * P_GRID) + approximate

        strategy.update()

        strategy.setObjective(step_cost, sense=GRB.MINIMIZE)
        strategy.update()
        strategy.optimize()
        S1 = strategy.ObjVal

        # Save results
        df_step_cost.loc[interval].step_cost = strategy.ObjVal

        # for v in strategy.getVars():
        #     print(' %s %g' % (v.varName, v.x))

        for v in strategy.getVars():
            df_decision_vars.loc[interval, '%s' % v.varName] = v.x

        Q_X1_sol = strategy.getAttr('x', Q_X1)
        Q_X2_sol = strategy.getAttr('x', Q_X2)
        Q_X3_sol = strategy.getAttr('x', Q_X3)
        Q_X4_sol = strategy.getAttr('x', Q_X4)
        Q_X5_sol = strategy.getAttr('x', Q_X5)
        Q_X6_sol = strategy.getAttr('x', Q_X6)
        Q_X7_sol = strategy.getAttr('x', Q_X7)

        for i in Intervals_CCGT:
            df_Q_X1.loc[(interval - 1) * 18 + i].Q_X1 = Q_X1_sol[i]
            df_Q_X2.loc[(interval - 1) * 18 + i].Q_X2 = Q_X2_sol[i]
            df_Q_X3.loc[(interval - 1) * 18 + i].Q_X3 = Q_X3_sol[i]
            df_Q_X4.loc[(interval - 1) * 18 + i].Q_X4 = Q_X4_sol[i]
            df_Q_X5.loc[(interval - 1) * 18 + i].Q_X5 = Q_X5_sol[i]
            df_Q_X6.loc[(interval - 1) * 18 + i].Q_X6 = Q_X6_sol[i]
            df_Q_X7.loc[(interval - 1) * 18 + i].Q_X7 = Q_X7_sol[i]
            df_Q_CCGT.loc[(interval - 1) * 18 + i].Q_CCGT = 0.4031 * Q_X1_sol[i] + 0.3656 * Q_X2_sol[i] + 0.06311 * \
                                                            Q_X3_sol[i] + 0.2087 * Q_X4_sol[i]

        df_P_CCGT.loc[interval].P_CCGT = g_CCGT.x * 16 - 10
        df_GRID_PRICE.loc[interval].GRID_PRICE = 1000 * price[interval] * P_GRID.x
        r_sol = strategy.getAttr('x', r)
        df_PLF_r.loc[interval].r = r_sol

        # Locate which slope d_a should be updated
        Q_update_a = (df_Q_CCGT.loc[interval * 18].Q_CCGT - 15) // ub + 1
        df_a.loc[interval].a = Q_update_a

        #  Obtain an observation of the marginal value(shadow price) by solve the numerical derivative
        change = strategy
        new1 = change.getConstrByName('new1')
        new1.setAttr('rhs', (-1 + df_Q_X2.loc[(interval - 1) * 18].Q_X2))
        new2 = change.getConstrByName('new2')
        new2.setAttr('rhs', (-1 + df_Q_X3.loc[(interval - 1) * 18].Q_X3))
        new3 = change.getConstrByName('new3')
        new3.setAttr('rhs', (-1 + df_Q_X4.loc[(interval - 1) * 18].Q_X4))
        new4 = change.getConstrByName('new4')
        new4.setAttr('rhs', (-1 + df_Q_X5.loc[(interval - 1) * 18].Q_X5))

        change.update()
        change.optimize()

        S2 = change.ObjVal
        # print(S1, S2)
        Partial = (S1 - S2) / (0.4031 + 0.3656 + 0.06311 + 0.2087)
        df_Partial.loc[interval].Partial = Partial

    # print(df_Partial)
    # print(df_a)

    Cost.append(df_step_cost['step_cost'].sum())


    ### Backward Update slopes
    from spar1 import spar_update

    b = 30       #tunable parameters
    step_size = b/(b+n-1)

    for interval in range(1,periods):
        a = df_a.loc[interval,'a']
        a_new=str(int(a))
        updated_slope = step_size * df_Partial.loc[interval+1,'Partial']+ (1-step_size) * df_slopes.loc[interval, a_new]
        # print(updated_slope)

        # update the new slope into df_slopes
        df_slopes.loc[interval, a_new] = updated_slope
        located_bars = df_slopes.loc[interval]
        # print(spar_update(located_bars, int(a)))
        df_slopes.loc[interval] = np.array(spar_update(located_bars, int(a)))

# print(df_slopes)
print(Cost)
plt.plot(Cost)
plt.show()

