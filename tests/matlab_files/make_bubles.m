N=256;%size of the image for inversion
Ntheta=64;%number of angular samples, not dense
Ns=N;%number of radial samples; Ns==N for this test;
ellipse = [1   .05  .05      0    0   0
           1    .04   .04    -0.5  -0.5 0
           1   .03   .03    -.3  -.3   0
          ]; 
[f,ellipse]=phantom(N,ellipse);filter_kind='hamming';%ramp,shepp-logan,cosine,cosine2,hamming,hann
ff=apply_filter_2d_exact(f,filter_kind,ellipse);ff=single(ff');
%filtered Radon data
h=apply_filter_exact(Ntheta,Ns,filter_kind,ellipse);h=single(h);

fid=fopen('../data/fbub','wb');
fwrite(fid,ff,'single');
fclose(fid);
fid=fopen('../data/Rbub','wb');
fwrite(fid,h,'single');
fclose(fid);
exit
