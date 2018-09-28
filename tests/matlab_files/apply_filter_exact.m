function [g,filter,t] = apply_filter_exact(Ntheta,Ns,filter,ellipse)

os=4;d=1/2;
th=linspace(0,pi,Ntheta+1);th=th(1:end-1);
Nse=os*(Ns);

Rhate=zeros(Nse,Ntheta);
for k=1:size(ellipse,1);
    Rhate = Rhate+fRphantom(os,Nse,th,ellipse(k,:),1-mod(Ns,2));
end;
% keyboard
t=(0:(Nse/2))/Nse;
switch filter
    case 'ramp'
        wfa=Nse*0.5*wint(12,t)';%.*(t/(2*d)<=1);%compute the weigths
        % Do nothing
    case 'shepp-logan'
        % be careful not to divide by 0:
        wfa = Nse*0.5*wint(12,t)'.*sinc(t/(2*d)).*(t/d<=2);
    case 'cosine'
        wfa = Nse*0.5*wint(12,t)'.*cos(pi*t./(2*d)).*(t/d<=1);
    case 'cosine2'
        wfa = Nse*0.5*wint(12,t)'.*(cos(pi*t./(2*d))).^2.*(t/d<=1); 
    case 'hamming'
        wfa = Nse*0.5*wint(12,t)'.*(.54 + .46 * cos(pi*t./d)).*(t/d<=1);
    case 'hann'
        wfa=Nse*0.5*wint(12,t)'.*(1+cos(pi*t./d)) / 2.*(t/d<=1);
    otherwise
        error(message('images:iradon:invalidFilter'))
end
wfa=wfa.*(wfa>=0);%.*(t/d<=1);
wfamid=2*wfa(1);%2*
wfa=[fliplr(wfa(2:end)) wfamid wfa(2:end)]';
wfa=wfa(1:end-1);

g=fftshift(ifft(ifftshift(diag(wfa)*Rhate)));%
g=g(Nse/2+1+(-floor(Ns/2):ceil(Ns/2)-1),:);
g=real(g');
filter=wfa;
t=[fliplr(-t(2:end)) 0 t(2:end-1)]';
end
