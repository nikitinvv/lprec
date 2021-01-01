/*interface*/
%module lpRgpu

%{
#define SWIG_FILE_WITH_INIT
#include "lpRgpu.cuh"
%}


class lpRgpu
{
public:
	%immutable;

	%mutable;	
	lpRgpu(size_t params, int gpu);
	~lpRgpu();
	void printGlobalParameters();
	void printFwdParameters();
	void printAdjParameters();
	void readGlobalParametersArr(float* params);
	void readFwdParametersArr(int* paramsi, float* paramsf);
	void readAdjParametersArr(int* paramsi, float* paramsf);
	void printCurrentGPUMemory(const char* str = 0);

	void initFwd(size_t paramsi, size_t paramsf, int gpu);
	void initAdj(size_t paramsi, size_t paramsf, int gpu);

	void deleteFwd();
	void deleteAdj();

	void prefilter2D(float *df, float* dtmpf,uint width, uint height);
	void execFwd();
	void execAdj();

	void execFwdManyPtr(size_t Rptr, size_t fptr, int Nslices0, int gpu);
	void execAdjManyPtr(size_t fptr, size_t Rpt, int Nslices0, int gpu);

	void applyFilter();
	void padding(int N_);
};
