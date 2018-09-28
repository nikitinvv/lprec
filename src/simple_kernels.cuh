//complex multiplication, y=C*y*x
__global__ void mul(float C, float2* y,float2* x,int N1,int N2, int ni)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=N1||ty>=N2||tz>=ni) return;
	float2 x0,y0;
	y0=y[tz*N1*N2+ty*N1+tx];x0=x[ty*N1+tx];
	y[tz*N1*N2+ty*N1+tx].x=C*(y0.x*x0.x-y0.y*x0.y);
	y[tz*N1*N2+ty*N1+tx].y=C*(y0.x*x0.y+y0.y*x0.x);
}

//division dR=dR/dgit
__global__ void pdiv (float* dgit,float* dR,int Ntheta_in,int Ns_in,int ni)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=Ntheta_in||ty>=Ns_in||tz>=ni) return;
	float t=dR[tz*Ntheta_in*Ns_in+ty*Ntheta_in+tx];
	if(t<1e-5) t=1e-5;
	dR[tz*Ntheta_in*Ns_in+ty*Ntheta_in+tx]=dgit[tz*Ntheta_in*Ns_in+ty*Ntheta_in+tx]/t;
}

//multiplication df=df*dfit
__global__ void pmul (float* df,float* dfit,int N1,int N2,int ni)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=N1||ty>=N2||tz>=ni) return;
	df[tz*N1*N2+ty*N1+tx]=df[tz*N1*N2+ty*N1+tx]*dfit[tz*N1*N2+ty*N1+tx];
}

//copy data with padding
__global__ void copyc(float* dR,float2* dRc,int N1,int N2, int os)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	if (tx>=N1||ty>=N2) return;
	int ind=ty*N1*os+tx+N1*os/2-(int)(N1/2);
	dRc[ind].x=dR[ty*N1+tx];
	dRc[ind].y=0;
}

//copy back data with padding
__global__ void copycback(float* dR,float2* dRc,int N1,int N2, int os)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	if (tx>=N1||ty>=N2) return;
	int ind=ty*N1*os+tx+N1*os/2-(int)(N1/2);
	dR[ty*N1+tx]=dRc[ind].x;
}

//standard fft shift
__global__ void fftshift(float2* dRc,int N1,int N2)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	if (tx>=N1||ty>=N2) return;
	int g=(1-2*((tx+1)%2))*(1-2*((ty+1)%2));
	dRc[ty*N1+tx].x=dRc[ty*N1+tx].x*g;
	dRc[ty*N1+tx].y=dRc[ty*N1+tx].y*g;
}

//multiplication by filter in frequency
__global__ void mulfilter(float2* dRc,float* dfilter,int N1,int N2)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	if (tx>=N1||ty>=N2) return;

	dRc[ty*N1+tx].x=dRc[ty*N1+tx].x*dfilter[tx];
	dRc[ty*N1+tx].y=dRc[ty*N1+tx].y*dfilter[tx];
}

__global__ void mulconst(float* dR,float c,int N1,int N2)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	if (tx>=N1||ty>=N2) return;
	dR[ty*N1+tx]=dR[ty*N1+tx]*c;
}
__global__ void circle(float* df,float* cir,int N1,int N2,int ni)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=N1||ty>=N2||tz>=ni) return;
	df[tz*N1*N2+ty*N1+tx]=df[tz*N1*N2+ty*N1+tx]*cir[ty*N1+tx];
}
__global__ void mcopy(float* df,float* dfit,int N1,int N2)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	if (tx>=N1||ty>=N2) return;
	df[ty*N1+tx]=dfit[ty*N1+tx];
}

__global__ void padker(float* R, int lb, int rb, int N1, int N2, int N3)
{
        uint tx = blockIdx.x*blockDim.x + threadIdx.x;
        uint ty = blockIdx.y*blockDim.y + threadIdx.y;
        uint tz = blockIdx.z*blockDim.z + threadIdx.z;

        if (tx>=N1||ty>=N2||tz>=N3) return;
        if (ty<lb) R[tz*N1*N2+ty*N1+tx]=R[tz*N1*N2+lb*N1+tx];
        if (ty>rb) R[tz*N1*N2+ty*N1+tx]=R[tz*N1*N2+rb*N1+tx];
}

