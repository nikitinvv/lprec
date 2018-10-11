/*interface*/
%module lpRgpu

%{
#define SWIG_FILE_WITH_INIT
#include "lpRgpu.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}


class lpRgpu{
	//global parameters
	int N;
	int Nspan;
	int Ntheta;int Nrho;
	int Nproj;
	int Nslices;int ni;
	int Ntheta_cut;
	int Ntheta_R2C;
	int cor;
	int osangles;
	float* erho;

	//grids storages
	glgrids* ggs;
	fwdgrids* fgs;
	adjgrids* ags;
	//gpu memory    
	float* derho;
	float* dfl;
	float2* dflc;
	cudaArray* dfla;
	cufftHandle plan_forward;
	cufftHandle plan_inverse;
	cufftHandle plan_f_forward;
	cufftHandle plan_f_inverse;
	//fwd
	float2* fZfwd;
	float2* dfZfwd;
	cudaArray* dfa;
	float* dR;
	float* dtmpf;

	//adj
	float2* fZadj;
	float2* dfZadj;
	float* dtmpR;
	cudaArray* dRa;
	float* df;

	//filter
    float* filter;
	float* dfilter;
	float2* dRc;
	
public:
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* params, int Nparams)};
	lpRgpu(float* params, int Nparams);
%clear (float* params, int Nparams);
	~lpRgpu();
	void printGlobalParameters();
	void printFwdParameters();
	void printAdjParameters();
	void readGlobalParametersArr(float* params);
	void readFwdParametersArr(int* paramsi, float* paramsf);
	void readAdjParametersArr(int* paramsi, float* paramsf);
	void printCurrentGPUMemory(const char* str = 0);

%apply (int* INPLACE_ARRAY1, int DIM1) {(int* paramsi, int Nparamsi)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* paramsf, int Nparamsf)};
	void initFwd(int* paramsi, int Nparamsi, float* paramsf, int Nparamsf);
%clear (int* paramsi, int Nparamsi);
%clear (float* paramsf, int Nparamsf);

%apply (int* INPLACE_ARRAY1, int DIM1) {(int* paramsi, int Nparamsi)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* paramsf, int Nparamsf)};
	void initAdj(int* paramsi, int Nparamsi, float* paramsf, int Nparamsf);
%clear (int* paramsi, int Nparamsi);
%clear (float* paramsf, int Nparamsf);

	void deleteFwd();
	void deleteAdj();

	void prefilter2D(float *df, float* dtmpf,uint width, uint height);
	void execFwd();
	void execAdj();

%apply (float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* R, int Nslices2_,int Nproj_, int N_)};
%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* f, int Nslices1_, int N2_, int N1_)};
	void execFwdMany(float* R, int Nslices2_, int Nproj_, int N_, float* f, int Nslices1_, int N2_, int N1_);
%clear (float* R, int Nslices2_,int Nproj_, int N_);
%clear (float* f, int Nslices1_, int N2_, int N1_);

%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* R, int Nslices2_,int Nproj_, int N_)};
%apply (float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* f, int Nslices1_, int N2_, int N1_)};
	void execAdjMany(float* f, int Nslices1_, int N2_, int N1_, float* R, int Nslices2_, int Nproj_, int N_);
%clear (float* R, int Nslices2_,int Nproj_, int N_);
%clear (float* f, int Nslices1_, int N2_, int N1_);

	void applyFilter();
	void padding(int N_);
};
