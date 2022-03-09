import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from pylab import rcParams

rcParams['figure.figsize'] = 11, 6
periods = 96

df_original_result = pd.read_csv("D:\\Nuts\CCGT\ADP\Deterministic\\0212ADP_penalty.csv")
P_wcur = Series(df_original_result['7'])
P_cur = Series(df_original_result['8'])
Q_cur = Series(df_original_result['11'])
Q_waste = Series(df_original_result['16'])
# CCGT_Dynamic.index = range(1, 97 )

df_equal =  pd.read_csv("D:\\Nuts\CCGT\ADP\Deterministic\\0212ADP_penalty_load_equal_waste.csv")
P_wcur_e = Series(df_equal['7'])
P_cur_e = Series(df_equal['8'])
Q_cur_e = Series(df_equal['11'])
Q_waste_e = Series(df_equal['16'])

df_less =  pd.read_csv("D:\\Nuts\CCGT\ADP\Deterministic\\0212ADP_penalty_load_less_waste.csv")
P_wcur_l = Series(df_less['7'])
P_cur_l = Series(df_less['8'])
Q_cur_l = Series(df_less['11'])
Q_waste_l = Series(df_less['16'])

df_zero =  pd.read_csv("D:\\Nuts\CCGT\ADP\Deterministic\\0212ADP_penalty_zero.csv")
P_wcur_z = Series(df_zero['7'])
P_cur_z = Series(df_zero['8'])
Q_cur_z = Series(df_zero['11'])
Q_waste_z = Series(df_zero['16'])

fig, ax = plt.subplots(1, 1, figsize=(11, 6))
fonten = {'family':'Times New Roman', 'size':15}

ax.set_xticks(range(1,97,5))
ax.set_xlim(-0.5,97)

plt.xlabel("Time Period (interval=15min)", fontsize=15, fontname='Times New Roman')
ax.set_ylabel(u'$P_{wcur}$ (MW)', fontdict=fonten)
# plt.title("Q_CCGT Contrast Between Stable and Dynamic Model",fontsize = 25,fontname='Times New Roman',y = 1.05)
plt.grid('minor', linestyle='-.')


plt.plot(P_wcur,  linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,350,350]', color = 'red', marker='v',markersize=5)
plt.plot(P_wcur_e, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,200,200]',color = 'blue', alpha=0.75, marker='v',markersize=5)
plt.plot(P_wcur_l, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,100,100]', color = 'darkgreen', alpha=0.75, marker='v',markersize=5)
plt.plot(P_wcur_z,  linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [10,10,10,10]', color = 'darkorange', marker='v',markersize=5)

# plt.plot(Q_waste, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,350,350]', color = 'red', marker='o',markersize=5)
# plt.plot(Q_waste_e, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,200,200]',color = 'blue', alpha=0.75, marker='o',markersize=5)
# plt.plot(Q_waste_l, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,100,100]', color = 'darkgreen', alpha=0.75, marker='o',markersize=5)
# plt.plot(Q_waste_z, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [10,10,10,10]', color = 'darkorange', marker='o',markersize=5)

# plt.plot(P_cur, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,350,350]', color = 'red', marker='*',markersize=5)
# plt.plot(P_cur_e, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,200,200]',color = 'blue', alpha=0.75, marker='*',markersize=5)
# plt.plot(P_cur_l, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,100,100]', color = 'darkgreen', alpha=0.75, marker='*',markersize=5)
# plt.plot(P_cur_z, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [10,10,10,10]', color = 'darkorange', marker='*',markersize=5)

# plt.plot(Q_cur, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,350,350]', color = 'red', marker='p',markersize=5)
# plt.plot(Q_cur_e, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,200,200]',color = 'blue', alpha=0.75, marker='p',markersize=5)
# plt.plot(Q_cur_l, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [200,200,100,100]', color = 'darkgreen', alpha=0.75, marker='p',markersize=5)
# plt.plot(Q_cur_z, linewidth = 1.5, label= '[$c_{wcur}, c_{qwte}, c_{els}, c_{hls}$] = [10,10,10,10]', color = 'darkorange', marker='p',markersize=5)


plt.legend(loc='upper left')
plt.savefig('D:\\Nuts\CCGT\ADP\Deterministic\\Penalty_P_wcur.jpeg',dpi=500,bbox_inches='tight')
plt.show()
