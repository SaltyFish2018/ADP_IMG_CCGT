function [ha, h] = barplotplot(x1,y1,x2,y2,x3,y3,x4,y4,xlim1, xlab, ylab)

ha(1) = axes('ycolor','k','yminortick','on','xminortick','off');
hold on;
h(1) = line(x1, y1,'parent',ha(1),'color','k');
if ~isempty(xlim1)
    set(ha(1), 'xlim', xlim1); 
end
xlim1 = get(ha(1),'xlim');
set(ha(1),'position',[0.15 0.15 0.65 0.8])
set(ha(1),'ylim',[2.3 3.3])
grid on
box on
% 画第二条线
pos1=get(ha(1),'position');
ha(2) = axes('position',pos1,'color','none','ycolor','g','yaxislocation','right','xlim',xlim1, ...
    'xtick', []);

h(2) = line(x2,y2,'color','g','parent',ha(2),'linewidth',1.2,'linestyle','-');
 set(ha(2),'ylim',[9.5 10.5])
% 画第三条线
pos1(1)=pos1(1)-0.02;
pos1(3) = pos1(3)*.86;
set([ha(1);ha(2)],'position',pos1);
pos3 = pos1;
pos3(3) = pos3(3)+.1;
xlim3 = xlim1;
xlim3(2) = xlim3(1)+1*(xlim1(2)-xlim1(1))/pos1(3)*pos3(3);
ha(3) = axes('position',pos3, 'color','none','ycolor','b','xlim',xlim3, ...
    'xtick',[],'yaxislocation','right','yminortick','on');
h(3) = line(x3, y3,'color','b','linewidth',1.2,'linestyle','-','parent',ha(3));
set(ha(3),'ylim',[27.5 29.5])
set(ha(3),'yTick',27.5:0.2:29.5)

ylim3 = get(ha(3), 'ylim');
line([xlim1(2),xlim3(2)],[ylim3(1),ylim3(1)],'parent',ha(3),'color','w');
% 画第四条线

pos4 = pos1;
pos4(3) = pos4(3)+.2;
xlim4 = xlim1;
xlim4(2) = xlim4(1)+1*(xlim1(2)-xlim1(1))/pos1(3)*pos4(3);
ha(4) = axes('position',pos4, 'color','none','ycolor','r','xlim',xlim4, ...
    'xtick',[],'yaxislocation','right','yminortick','on');
h(4) = line(x4, y4,'color','r','linewidth',1.2,'linestyle','-','parent',ha(4));
set(ha(4),'ylim',[48.5 50.5])
set(ha(4),'yTick',48.5:0.2:50.5)
% 隐藏第三个横轴伸出来的部分
ylim4 = get(ha(4), 'ylim');
line([xlim1(2),xlim4(2)],[ylim4(1),ylim4(1)],'parent',ha(4),'color','w');

% 加ylabels
hylab = get([ha(1);ha(2);ha(3);ha(4)],'ylabel');
hxlab = get(ha(1),'xlabel');
% set(hylab{1},'string',ylab{1});
% set(hylab{2},'string',ylab{2});
% set(hylab{3},'string',ylab{3});
% set(hylab{4},'string',ylab{4});
set(hxlab,'string', xlab);
hold off;
end
