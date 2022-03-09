import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from pylab import rcParams


rcParams['figure.figsize'] = 9, 6

adp_test_cost = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\adp_test_cost.npy')
myopic_test_cost = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\myopic_test_cost.npy')
ground_truth = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\milp_test_cost.npy')

adp_error = list(map(lambda x:(x[1]-x[0])/x[0],zip(ground_truth,adp_test_cost)))
myopic_error = list(map(lambda x:(x[1]-x[0])/x[0],zip(ground_truth,myopic_test_cost)))



x = [0]
i = 0
while i<=0.07:
    i+=0.0005
    x.append(i)

plt.hist(adp_error,x,histtype='bar',label='VFA-ADP',rwidth=0.8)
plt.hist(myopic_error,x,histtype='bar',label='Myopic',rwidth=0.8)

# plt.legend()
# plt.xlabel('Real-time optimization error', fontsize = 15, fontname='Times New Roman')
# plt.ylabel('Probability', fontsize = 15, fontname='Times New Roman')
# plt.grid(axis="y", linestyle='-.')
# plt.savefig('D:\\Nuts\CCGT\ADP\Stochastic\\test_error_distribution.jpeg',dpi=500,bbox_inches='tight')
# plt.show()

adp_test_cost_1 = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\adp_robust_cost_e1.npy')
ground_truth_1 = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\milp_robust_cost_e1.npy')
adp_error_1 = list(map(lambda x:(x[1]-x[0])/x[0],zip(ground_truth_1,adp_test_cost_1)))

adp_test_cost_2 = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\adp_robust_cost_e2.npy')
ground_truth_2 = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\milp_robust_cost_e2.npy')
adp_error_2 = list(map(lambda x:(x[1]-x[0])/x[0],zip(ground_truth_2,adp_test_cost_2)))

adp_test_cost_3 = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\adp_robust_cost_e3.npy')
ground_truth_3 = np.load('D:\\Nuts\CCGT\ADP\Stochastic\\milp_robust_cost_e3.npy')
adp_error_3 = list(map(lambda x:(x[1]-x[0])/x[0],zip(ground_truth_3,adp_test_cost_3)))

print(np.mean(adp_error))
print(np.mean(adp_error_1))
print(np.mean(adp_error_2))
# print(np.mean(adp_error_3))

fig,axes = plt.subplots()
axes.boxplot(x=adp_error,sym='rd',positions=[0.01],widths=0.005)
axes.boxplot(x=adp_error_1,sym='rd',positions=[0.02],widths=0.005)
axes.boxplot(x=adp_error_2,sym='rd',positions=[0.03],widths=0.005)
# axes.boxplot(x=adp_error_3,sym='rd',positions=[0.04],widths=0.005)

# 把y轴的主刻度设置为0.005的倍数
y_major_locator=MultipleLocator(0.005)
axes.yaxis.set_major_locator(y_major_locator)

plt.xlim(0,0.04)
plt.ylim(0,0.02)
plt.xlabel(r'$\delta_{p}$', fontsize = 15, fontname='Times New Roman')
plt.ylabel('Optimization error', fontsize = 20, fontname='Times New Roman')
plt.grid(axis="y", linestyle='--')
# plt.savefig('D:\\Nuts\CCGT\ADP\Stochastic\\robust_p.png',dpi=500,bbox_inches='tight')

plt.show()