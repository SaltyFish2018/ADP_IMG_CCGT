

def smooth_update(bars):

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env()
    env.start()

    self = gp.Model('smooth')

    lb = -10
    x1 = self.addVar(vtype=GRB.CONTINUOUS, lb=lb, name='x1')
    x2 = self.addVar(vtype=GRB.CONTINUOUS, lb=lb, name='x2')
    x3 = self.addVar(vtype=GRB.CONTINUOUS, lb=lb, name='x3')
    x4 = self.addVar(vtype=GRB.CONTINUOUS, lb=lb, name='x4')
    x5 = self.addVar(vtype=GRB.CONTINUOUS, lb=lb, name='x5')

    self.update()

    self.addConstr(x1 <= x2)
    self.addConstr(x2 <= x3)
    self.addConstr(x3 <= x4)
    self.addConstr(x4 <= x5)
    self.update()

    # 定义目标函数
    distance = (x1 - bars[0])*(x1- bars[0])\
               + (x2 - bars[1])*(x2 - bars[1]) \
               + (x3 - bars[2])*(x3 - bars[2])\
               + (x4 - bars[3])*(x4 - bars[3]) \
               + (x5 - bars[4])*(x5 - bars[4])
    self.update()

    self.setObjective(distance, sense=GRB.MINIMIZE)
    self.update()
    self.setParam('OutputFlag', 0)
    self.optimize()


    bars[0] = x1.x
    bars[1] = x2.x
    bars[2] = x3.x
    bars[3] = x4.x
    bars[4] = x5.x

    return bars