#include"math_operators.cuh"
#define Pole -0.267949192431123f//(sqrt(3.0f)-2.0f)  //pole for cubic b-spline

texture<float, cudaTextureType2DLayered, cudaReadModeElementType> texf; //type 0
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> texR; //type 1
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> texfl; //type 2

//compute weigts for the cubic B spline
__host__ __device__ void bspline_weights(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
{
	const float2 one_frac = 1.0f - fraction;
	const float2 squared = fraction * fraction;
	const float2 one_sqd = one_frac * one_frac;

	w0 = 1.0f/6.0f * one_sqd * one_frac;
	w1 = 2.0f/3.0f - 0.5f * squared * (2.0f-fraction);
	w2 = 2.0f/3.0f - 0.5f * one_sqd * (2.0f-one_frac);
	w3 = 1.0f/6.0f * squared * fraction;
}

__device__ float linearTex2D(texture<float, cudaTextureType2DLayered, cudaReadModeElementType> tex, float x, float y, float z, int N0,int N1)
{
	float2 t0;
	t0.x = x/(float)N0;
	t0.y = y/(float)N1;
	return tex2DLayered(tex, t0.x, t0.y, z);
}

//cubic interpolation via two linear interpolations for several slices, texture is not normalized
__device__ float cubicTex2D(texture<float, cudaTextureType2DLayered, cudaReadModeElementType> tex, float x, float y, float z, int N0,int N1)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
	const float2 index = floor(coord_grid);
	const float2 fraction = coord_grid - index;
	float2 w0, w1, w2, w3;
	bspline_weights(fraction, w0, w1, w2, w3);

	const float2 g0 = w0 + w1;
	const float2 g1 = w2 + w3;
	const float2 h0 = (w1 / g0) - make_float2(0.5f) + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const float2 h1 = (w3 / g1) + make_float2(1.5f) + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

	float2 t0,t1;
	t0.x = h0.x/(float)N0;
	t1.x = h1.x/(float)N0;
	t0.y = h0.y/(float)N1;
	t1.y = h1.y/(float)N1;
	float tex00 = tex2DLayered(tex, t0.x, t0.y, z);
	float tex10 = tex2DLayered(tex, t1.x, t0.y, z);
	float tex01 = tex2DLayered(tex, t0.x, t1.y, z);
	float tex11 = tex2DLayered(tex, t1.x, t1.y, z);


	// weigh along the y-direction
	tex00 = g0.y * tex00 + g1.y * tex01;
	tex10 = g0.y * tex10 + g1.y * tex11;

	// weigh along the x-direction
	return (g0.x * tex00 + g1.x * tex10);
}

//interp from Cartesian to log-polar grid
__global__ void interp(int interp_id, float *fo, float* x, float* y, int W, int Np,int N1,int N2, int ni, int* cids, int step2d)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	uint tid = ty*W+tx;
	if(tid>=Np||tz>=ni) return;
	float u = x[tid]+0.5f;
	float v = y[tid]+0.5f;
	
	switch(interp_id)//no overhead, all threads have the same way
	{ 		
		case 0: fo[tz*step2d+cids[tid]] += linearTex2D(texf, u, v, tz,N1,N2);break;
		case 1: fo[tz*step2d+cids[tid]] += linearTex2D(texR, u, v, tz,N1,N2);break;   
		case 2: fo[tz*step2d+cids[tid]] += linearTex2D(texfl, u, v, tz,N1,N2);break;   
		case 3: fo[tz*step2d+cids[tid]] += cubicTex2D(texf, u, v, tz,N1,N2);break;
		case 4: fo[tz*step2d+cids[tid]] += cubicTex2D(texR, u, v, tz,N1,N2);break;   
		case 5: fo[tz*step2d+cids[tid]] += cubicTex2D(texfl, u, v, tz,N1,N2);break;   

	}	
}

//multiplication by e^\rho
__global__ void mulexp(float* y,float* x,int N1,int N2,int ni)//y = y*exp(x)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=N1||ty>=N2||tz>=ni) return;
	y[tz*N1*N2+ty*N1+tx]*=x[ty*N1+tx];
}

//casual cofficients for prefilter
__host__ __device__ float InitialCausalCoefficient(float* c, uint DataLength,int step)
{
	const uint Horizon = 12<DataLength?12:DataLength;

	// this initialization corresponds to clamping boundaries
	// accelerated loop
	float zn = Pole;
	float Sum = *c;
	for (uint n = 0; n < Horizon; n++) {
		Sum += zn * *c;
		zn *= Pole;
		c = (float*)((unsigned char*)c + step);
	}
	return(Sum);
}

//anticasual coffeicients for prefilter
__host__ __device__ float InitialAntiCausalCoefficient(float* c,uint DataLength,int step)
{
	// this initialization corresponds to clamping boundaries
	return((Pole / (Pole - 1.0f)) * *c);
}

//compute coefficients from samples c
__host__ __device__ void ConvertToInterpolationCoefficients(float* coeffs,uint DataLength,int step)
{
	// compute the overall gain
	const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);

	// causal initialization
	float* c = coeffs;
	float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
	*c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
	// causal recursion
	for (uint n = 1; n < DataLength; n++) {
		c = (float*)((unsigned char*)c + step);
		*c = previous_c = Lambda * *c + Pole * previous_c;
	}
	// anticausal initialization
	*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
	// anticausal recursion
	for (int n = DataLength - 2; 0 <= n; n--) {
		c = (float*)((unsigned char*)c - step);
		*c = previous_c = Pole * (previous_c - *c);
	}
}

//fast transpose on GPU
__global__ void transpose(float *odata, float *idata, int width, int height, int ni)
{
	__shared__ float block[MBS33][MBS31][MBS31+1];

	// read the matrix tile into shared memory
	uint xIndex = blockIdx.x * MBS31 + threadIdx.x;
	uint yIndex = blockIdx.y * MBS32 + threadIdx.y;
	uint zIndex = blockIdx.z*blockDim.z + threadIdx.z;
	if(zIndex>=ni) return;
	if((xIndex < width) && (yIndex < height))
	{
		uint index_in = zIndex*width*height+yIndex * width + xIndex;
		block[threadIdx.z][threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * MBS31 + threadIdx.x;
	yIndex = blockIdx.x * MBS32 + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		uint index_out = zIndex*width*height+yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.z][threadIdx.x][threadIdx.y];
	}
}

// compute coefficients from samples only for rows (gpu cache optimization) 
__global__ void SamplesToCoefficients2DY(float* image,uint pitch,uint width,uint height, int ni)
{
	// process lines in x-direction
	uint yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	if(yIndex>=ni) return;

	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x>=width) return;

	float* line = yIndex*width*height+image + x;  //direct access

	ConvertToInterpolationCoefficients(line, height, pitch);
}

