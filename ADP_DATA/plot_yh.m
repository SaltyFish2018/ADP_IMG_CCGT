
close all;

% wind = xlsread('wind_turbine.xls');
% figure;
% plot(wind(:,2), 'LineWidth', 2);
% grid on;
% grid minor;
% xlabel('Time Period', 'FontWeight','bold','FontSize',14);
% ylabel('P_{wt} (MW)', 'FontWeight','bold','FontSize',14);
% legend('Ԥ�����');
% set(gca,'FontSize',12,'Fontwei','Bold','Linewidth',1,'GridAlpha',.5)
% 
% figure;
% plot(wind(:,2), 'LineWidth', 2);
% hold on;
% plot(wind(:,3), 'LineWidth', 2);
% grid on;
% grid minor;
% xlabel('Time Period', 'FontWeight','bold','FontSize',14);
% ylabel('P_{wt} (MW)', 'FontWeight','bold','FontSize',14);
% legend('Ԥ�����', 'ʵ�����');
% set(gca,'FontSize',12,'Fontwei','Bold','Linewidth',1,'GridAlpha',.5)

% elec = xlsread('forcasted_electricity_demand_continous.xls');
% heat = xlsread('forcasted_heat_demand.xls')
% 
% figure;
% plot(elec(:,2), 'LineWidth', 2);
% hold on;
% plot(heat(:,4),'LineWidth', 2)
% grid on;
% grid minor;
% xlabel('Time Period', 'FontWeight','bold','FontSize',14);
% ylabel('D (MW)', 'FontWeight','bold','FontSize',14);
% ylim([10,65])
% legend('�縺��Ԥ��','�ȸ���Ԥ��');
% set(gca,'FontSize',12,'Fontwei','Bold','Linewidth',1,'GridAlpha',.5)

price = xlsread('forcasted_elec_price.xls');
figure;
plot(price(:,2), 'LineWidth', 2,'color','black');
grid on;
grid minor;
xlabel('Time Period', 'FontWeight','bold','FontSize',14);
ylabel('Price ($/KWh)', 'FontWeight','bold','FontSize',14);
ylim([0.04,0.11])
legend('Ԥ���г����');
set(gca,'FontSize',12,'Fontwei','Bold','Linewidth',1,'GridAlpha',.5)

