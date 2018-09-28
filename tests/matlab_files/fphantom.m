function fhate2 = fRphantom(os,Ne,P,doshift)
A=P(1);
a=P(2);asq=a*a;
b=P(3);bsq=b*b;
x0=P(4);
y0=P(5);
phi=P(6)*pi/180;


[k1,k2]=ndgrid((-Ne/2:(Ne/2))/2/os,(-Ne/2:(Ne/2))/2/os);
k1h=cos(phi).*k1-sin(phi).*k2;
k2h=sin(phi).*k1+cos(phi).*k2;
K=sqrt((a*k1h).^2+(b*k2h).^2);
fhate1=A*a*b*pi*besselj(1,2*pi*K)./(pi*K).* exp(-2*pi*1i*(k1*(-x0)+k2*(y0)));%-x0, y0 (matlab feature)
fhate1((end+1)/2,(end+1)/2)=A*a*b*pi*1;%(1-(pi*K).^2+(pi*K).^4/12);%approx Taylor
fhate1=fhate1*(Ne/os/2)^2;
if doshift==1,
    fhate1=fhate1.*exp(-2*pi*1i*(1*os/Ne*k1+1*os/Ne*k2));
end;
% fhate2=[flipud(fliplr(fhate1(2:end,:))) flipud(fhate1(2:end,2:end));fliplr(fhate1(:,:)) fhate1(:,2:end)];
fhate2=fhate1(1:end-1,1:end-1);
% fhate2=fhate1;
end
