% ��һ��bar������
sample=1200;
sn=0.002;
x1=stepdata(1:sample,1);
y1=stepdata(1:sample,2);
% �ڶ���line������
x2=stepdata(1:sample,1);
y2=stepdata(1:sample,5);
noise=filter([1],[1 -.8],randn(sample,1));
noise2=noise/std(noise)*sqrt(sn)*std(y2);
y2=y2+noise2;
% ������line������
x3=stepdata(1:sample,1);
y3=stepdata(1:sample,3);
noise3=noise/std(noise)*sqrt(sn)*std(y3);
y3=y3+noise3;
% ���ĸ�line������
x4=stepdata(1:sample,1);
y4=stepdata(1:sample,4);
noise4=noise/std(noise)*sqrt(sn)*std(y3);
y4=y4+noise4;
figure
[ha,h] = barplotplot(x1,y1,x2,y2,x3,y3,x4,y4,[],...
    {'Time(s)'},{'m_{gas}(kg/s)','E_{ST}(MW)','E_{GT}(MW)','Q_{DH}(MW)'}); % xlim����ָ���� % Ϊ[]��ʾ����Ĭ��ֵ
