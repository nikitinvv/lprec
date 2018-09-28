function g1 = fRphantom(os,Nse,th,P,doshift)
% keyboard
s=(-Nse/2:(Nse/2)-1)/2/os;
% s=linspace(-Nse/4/os,Nse/4/os,Nse);
A=P(1);
a=P(2);asq=a*a;
b=P(3);bsq=b*b;
x0=P(4);
y0=P(5);
phi=-P(6)*pi/180;

Nrho=4*Nse;
% rhosp=(-Nrho/2:(Nrho/2)-1)/Nrho*2;rho=rhosp;
rhosp=(-Nrho/2:(Nrho/2)-1)/Nrho*2;rho=rhosp;
% rhosp=linspace(-1,1,Nrho);rho=rhosp;
% thsp=linspace(0,pi,Ntheta);theta=thsp;
theta=th;
Ntheta=numel(th);
rho0=sqrt(asq*(cos(theta-phi)).^2+bsq*(sin(theta-phi)).^2);
g1=zeros(Ntheta,Nse);
rhop=rho(:);
ep=exp(-2*pi*1i*rhop*s)';
for it=1:Ntheta
    idx=abs(rhop+x0*cos(theta(it))-y0*sin(theta(it)))<=rho0(it);
    ss1=2*A*sqrt(asq*bsq)*sqrt(rho0(it).^2-(rhop+x0*cos(theta(it))-y0*sin(theta(it))).^2)./(rho0(it).^2);
    ss=ep*(idx.*ss1);
    g1(it,:)=g1(it,:)+ss';
end
g1=g1'*(rhosp(2)-rhosp(1))*Nse/2/os;
% keyboard
if doshift==1,
    g1=g1.*repmat(exp(-2*pi*i*(4*os/Nrho)*s)',[1 Ntheta]);
end;