function w=wint(n,t)
% keyboard
N=length(t);
s=linspace(1e-40,1,n)';
iv=inv(exp([0:n-1]'*log(s'))');%Inverse vandermonde matrix
u=diff(exp([1:n+1]'*log(s'))'.*repmat(1./(1:n+1),n,1)); %integration over short intervals
W1=u(:,2:n+1)*iv;%x*pn(x) term
W2=u(:,1:n)*iv;%const*pn(x) term

p=1./[1:n-1 (n-1)*ones(1,N-2*(n-1)-1) n-1:-1:1]';%Compensate for overlapping short intervals
w=zeros(N,1);
for j=1:N-n+1,
  W=(t(j+n-1)-t(j))^2*W1+(t(j+n-1)-t(j))*t(j)*W2;%Change coordinates, and constant and linear parts
  for k=1:n-1,
    w(j+(0:n-1))=w(j+(0:n-1))+W(k,:)'*p(j+k-1);% Add to weights
  end;
 end;
 wn=w;
 wn(N-40+1:N)=(w(N-40+1)/(N-40))*(N-40:N-1);
 w=wn;
