import sys
import datetime
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from pylab import rcParams

rcParams['figure.figsize'] = 11, 6
periods = 96

df_CCGT_Dynamic  = pd.read_csv("D:\\Nuts\CCGT\MILP\\0915Qccgt.csv")
CCGT_Dynamic = Series(df_CCGT_Dynamic['Q_CCGT'])
CCGT_Dynamic.index = range(18, 18*periods + 1,18 )

df_CCGT_Stable =  pd.read_csv("D:\\Nuts\CCGT\MILP\\0916Qccgt_stable.csv")
CCGT_Stable = Series(df_CCGT_Stable['Q_CCGT'])
CCGT_Stable.index = range(18, 18*periods + 1, 18)

df_CCGT_Dynamic_all = pd.read_csv("D:\\Nuts\CCGT\MILP\\0915Qccgt_all.csv")
CCGT_Dynamic_all = Series(df_CCGT_Dynamic_all['Q_CCGT'])
CCGT_Dynamic_all.index = range(1, 18*periods + 1)

fig, ax = plt.subplots(1, 1, figsize=(11, 6))

# plt.xticks(fontsize=15, fontname='Times New Roman')
# plt.yticks(fontsize=15, fontname='Times New Roman')
plt.xlabel("Time Period (interval=50s)", fontsize=15, fontname='Times New Roman')
plt.ylabel("$Q_{CCGT}$ (MW)", fontsize=15, fontname='Times New Roman')
# plt.title("Q_CCGT Contrast Between Stable and Dynamic Model",fontsize = 25,fontname='Times New Roman',y = 1.05)
plt.grid(linestyle='-.')

plt.scatter(x = CCGT_Stable.index, y = CCGT_Stable, color = 'blue', marker='x',s=10)
plt.plot(CCGT_Stable, color = 'blue', linewidth = 1.5, label= 'Case1: Stable')

plt.scatter(x = CCGT_Dynamic.index, y = CCGT_Dynamic, color = 'red',s=10)
plt.plot(CCGT_Dynamic_all[18:], color = 'red', linewidth = 1.5, label= 'Case2: Dynamic',alpha=0.5)


# x1 = CCGT_Stable.index
# x1_new = np.linspace(18,18*periods,96)
# CCGT_Stable_smooth = make_interp_spline(x1,CCGT_Stable)(x1_new)
# plt.plot(x1_new, CCGT_Stable_smooth, color = 'blue', linewidth = 1.5, label= 'stable')

plt.legend()

# plot the magnified subfigures
axins = inset_axes(ax, width="30%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.5, 0.05, 1, 1),
                   bbox_transform=ax.transAxes)
#loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
axins.plot(CCGT_Dynamic_all, color = 'red', linewidth = 0.5)
axins.plot(CCGT_Stable, color = 'blue', linewidth = 0.5)
axins.scatter(x = CCGT_Dynamic.index, y = CCGT_Dynamic, color = 'red',s=10)
axins.scatter(x = CCGT_Stable.index, y = CCGT_Stable, color = 'blue', marker='x',s=10)


# 设置放大区间
zone_left = 70
zone_right = 72

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5 # x轴显示范围的扩展比例
y_ratio = 0.5 # y轴显示范围的扩展比例

# X轴的显示范围
x = CCGT_Stable.index
xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((CCGT_Dynamic_all[18*zone_left:18*zone_right], CCGT_Dynamic[18*zone_left:18*zone_right], CCGT_Stable[18*zone_left:18*zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

# 画两条线
xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

plt.grid(linestyle='-.')

plt.savefig("D:\\Nuts\CCGT\MILP\\09Contrast between Dynamic and Stable.jpeg",dpi=500,bbox_inches='tight')
plt.show()



