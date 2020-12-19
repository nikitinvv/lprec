#include"gridStorage.cuh"
//storage for Cartesian, polar and log-polar grids,
//functions to work with 2D data
//gpu
void gridStorage::init2dfloatgpu(float **&tmpa,float** &da,float **a,int* Np,int c, int Nsp)
{
	err=cudaMalloc((void**)&tmpa, Nsp * sizeof(float*));if (err!=0) callErr(cudaGetErrorString(err));
	da=new float*[Nsp];
	for(int i=0; i<Nsp; i++) {err=cudaMalloc(&da[i], c*Np[i]*sizeof(float)); if (err!=0) callErr(cudaGetErrorString(err));}
	for(int i=0; i<Nsp; i++) {err=cudaMemcpy(da[i],a[i],c*Np[i]*sizeof(float),cudaMemcpyHostToDevice); if (err!=0) callErr(cudaGetErrorString(err));} 
	err=cudaMemcpy(tmpa, da, Nsp*sizeof(float *), cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
}
void gridStorage::init2dfloatgpu(float **&tmpa,float** &da,float **a,int Np,int c, int Nsp)
{
	err=cudaMalloc((void**)&tmpa, Nsp * sizeof(float*));if (err!=0) callErr(cudaGetErrorString(err));
	da=new float*[Nsp];
	for(int i=0; i<Nsp; i++) {err=cudaMalloc(&da[i], c*Np*sizeof(float)); if (err!=0) callErr(cudaGetErrorString(err));}
	for(int i=0; i<Nsp; i++) {err=cudaMemcpy(da[i],a[i],c*Np*sizeof(float),cudaMemcpyHostToDevice); if (err!=0) callErr(cudaGetErrorString(err));}
	err=cudaMemcpy(tmpa, da, Nsp*sizeof(float *), cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
}
void gridStorage::init2dintgpu(int **&tmpa,int** &da,int **a,int* Np,int c, int Nsp)
{
	err=cudaMalloc((void**)&tmpa, Nsp * sizeof(int*));if (err!=0) callErr(cudaGetErrorString(err));
	da=new int*[Nsp];
	for(int i=0; i<Nsp; i++) {err=cudaMalloc(&da[i], c*Np[i]*sizeof(int)); if (err!=0) callErr(cudaGetErrorString(err));}
	for(int i=0; i<Nsp; i++) {err=cudaMemcpy(da[i],a[i],c*Np[i]*sizeof(float),cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));} 
	err=cudaMemcpy(tmpa, da, Nsp*sizeof(float *), cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
}
void gridStorage::init2dintgpu(int **&tmpa,int** &da,int **a,int Np,int c, int Nsp)
{
	err=cudaMalloc((void**)&tmpa, Nsp * sizeof(int*));if (err!=0) callErr(cudaGetErrorString(err));
	da=new int*[Nsp];
	for(int i=0; i<Nsp; i++) {err=cudaMalloc(&da[i], c*Np*sizeof(int)); if (err!=0) callErr(cudaGetErrorString(err));}
	for(int i=0; i<Nsp; i++) {err=cudaMemcpy(da[i],a[i],c*Np*sizeof(float),cudaMemcpyHostToDevice); if (err!=0) callErr(cudaGetErrorString(err));}
	err=cudaMemcpy(tmpa, da, Nsp*sizeof(float *), cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
}
void gridStorage::delete2dfloatgpu(float** tmpa,float** da, int Nsp)
{
	err=cudaMemcpy(da, tmpa, Nsp*sizeof(float *), cudaMemcpyDeviceToHost);if (err!=0) callErr(cudaGetErrorString(err));
	for(int i=0; i<Nsp; i++)
	{
		cudaFree(da[i]);
	}
	cudaFree(tmpa);
}
void gridStorage::delete2dintgpu(int** tmpa,int** da, int Nsp)
{ 
	err=cudaMemcpy(da, tmpa, Nsp*sizeof(int *), cudaMemcpyDeviceToHost);if (err!=0) callErr(cudaGetErrorString(err));
	for(int i=0; i<Nsp; i++)
	{
		cudaFree(da[i]);
	}
	cudaFree(tmpa);
}

//cpu
void gridStorage::delete2dfloat(float** a,int Ns)
{
	for(int i=0;i<Ns;i++) delete[] a[i];
	delete[] a;
}
void gridStorage::delete2dint(int** a,int Ns)
{
	for(int i=0;i<Ns;i++) delete[] a[i];
	delete[] a;
}

fwdgrids::fwdgrids(int Nspan_)
{
	Nspan=Nspan_;
}
adjgrids::adjgrids(int Nspan_)
{
	Nspan=Nspan_;
}
void fwdgrids::initgpu()
{	
	err=cudaMalloc((void **)&dcidsfwd, Ncidsfwd*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));

	//init cutted Cartesian and log-polar grids
	init2dfloatgpu(tlp2C1,dlp2C1,lp2C1,Ncidsfwd,1,Nspan);
	init2dfloatgpu(tlp2C2,dlp2C2,lp2C2,Ncidsfwd,1,Nspan);
	init2dfloatgpu(tp2lp1,dp2lp1,p2lp1,Npids,1,Nspan);
	init2dfloatgpu(tp2lp2,dp2lp2,p2lp2,Npids,1,Nspan);
	init2dintgpu(tpids,dpids,pids,Npids,1,Nspan);
	//copy indeces for grids
	err=cudaMemcpy(dcidsfwd,cidsfwd,Ncidsfwd*sizeof(int),cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
}
void adjgrids::initgpu()
{
	err=cudaMalloc((void **)&dcidsadj, Ncidsadj*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dlpidsadj, Nlpidsadj*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dwids, Nwids*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));

	//init cutted log-polar and 2 polar grids
	init2dfloatgpu(tC2lp1,dC2lp1,C2lp1,Ncidsadj,1,Nspan);
	init2dfloatgpu(tC2lp2,dC2lp2,C2lp2,Ncidsadj,1,Nspan);
	init2dfloatgpu(tlp2p1,dlp2p1,lp2p1,Nlpidsadj,1,Nspan);
	init2dfloatgpu(tlp2p2,dlp2p2,lp2p2,Nlpidsadj,1,Nspan);
	init2dfloatgpu(tlp2p1w,dlp2p1w,lp2p1w,Nwids,1,Nspan);
	init2dfloatgpu(tlp2p2w,dlp2p2w,lp2p2w,Nwids,1,Nspan);

	//copy indeces for grids
	err=cudaMemcpy(dcidsadj,cidsadj,Ncidsadj*sizeof(int),cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(dlpidsadj,lpidsadj,Nlpidsadj*sizeof(int),cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(dwids,wids,Nwids*sizeof(int),cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));
}

fwdgrids::~fwdgrids()
{
	//delete arrays allocated in readFwdParameters
	delete2dfloat(lp2C1,Nspan);
	delete2dfloat(lp2C2,Nspan);
	delete2dfloat(p2lp1,Nspan);
	delete2dfloat(p2lp2,Nspan);
	delete2dint(pids,Nspan);  

	delete[] cidsfwd;
	delete[] Npids;

	//free gpu memory
	delete2dfloatgpu(tlp2C1,dlp2C1,Nspan);
	delete2dfloatgpu(tlp2C2,dlp2C2,Nspan);
	delete2dfloatgpu(tp2lp1,dp2lp1,Nspan);
	delete2dfloatgpu(tp2lp2,dp2lp2,Nspan);
	delete2dintgpu(tpids,dpids,Nspan);
	cudaFree(dcidsfwd);

}
adjgrids::~adjgrids()
{
	//delete arrays allocated in readAdjParameters
	delete2dfloat(C2lp1,Nspan);
	delete2dfloat(C2lp2,Nspan);
	delete2dfloat(lp2p1,Nspan);
	delete2dfloat(lp2p2,Nspan);
	delete2dfloat(lp2p1w,Nspan);
	delete2dfloat(lp2p2w,Nspan);
	delete[] cidsadj;
	delete[] lpidsadj;
	delete[] wids;
	//free gpu memory
	delete2dfloatgpu(tC2lp1,dC2lp1,Nspan);
	delete2dfloatgpu(tC2lp2,dC2lp2,Nspan);
	delete2dfloatgpu(tlp2p1,dlp2p1,Nspan);
	delete2dfloatgpu(tlp2p2,dlp2p2,Nspan);
	delete2dfloatgpu(tlp2p1w,dlp2p1w,Nspan);
	delete2dfloatgpu(tlp2p2w,dlp2p2w,Nspan);
	cudaFree(dcidsadj);
	cudaFree(dlpidsadj);
	cudaFree(dwids);

}
