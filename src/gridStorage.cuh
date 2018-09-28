#pragma once
#include"config.cuh"

class gridStorage
{	
public:
	cudaError_t err;
	int Nspan;
	void init2dfloatgpu(float **&tmpa,float** &da,float **a,int* Np,int c, int Nsp);
	void init2dintgpu(int **&tmpa,int** &da,int **a,int* Np,int c, int Nsp);
	void init2dfloatgpu(float **&tmpa,float** &da,float **a,int Np,int c, int Nsp);
	void init2dintgpu(int **&tmpa,int** &da,int **a,int Np,int c, int Nsp);
	void delete2dfloatgpu(float** tmpa,float** da, int Nsp);
	void delete2dintgpu(int** tmpa,int** da, int Nsp);
	void delete2dfloat(float** a,int Ns);
	void delete2dint(int** a,int Ns);
};

class fwdgrids: public gridStorage
{	
public:
	float** lp2C1;
	float** lp2C2;
	float** p2lp1;
	float** p2lp2;
	int* cidsfwd;
	int** pids;
	float** tlp2C1;
	float** tlp2C2;
	float** tp2lp1;
	float** tp2lp2;
	int** tpids;
	//gpu
	float** dlp2C1;
	float** dlp2C2;
	float** dp2lp1;
	float** dp2lp2;
	int** dpids;

	int* dcidsfwd;
	int Ncidsfwd;
	int* Npids;

	void initgpu();
	fwdgrids(int Nspan);
	~fwdgrids();
};


class adjgrids: public gridStorage
{
public:
	float** C2lp1;
	float** C2lp2;
	float** lp2p1;
	float** lp2p2;
	float** lp2p1w;
	float** lp2p2w;
	int* cidsadj;
	int* lpidsadj;
	int* wids;
	float** tC2lp1;
	float** tC2lp2;
	float** tlp2p1;
	float** tlp2p2;
	float** tlp2p1w;
	float** tlp2p2w;

	//gpu memory
	float** dC2lp1;
	float** dC2lp2;
	float** dlp2p1;
	float** dlp2p2;
	float** dlp2p1w;
	float** dlp2p2w;
	int* dcidsadj;
	int* dlpidsadj;
	int* dwids;
	int Ncidsadj;
	int Nlpidsadj;
	int Nwids;

	void initgpu();
	adjgrids(int Nspan);
	~adjgrids();
};
