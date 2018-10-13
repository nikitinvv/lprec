#include"lpRgpu.cuh"
#include<stdio.h>

void lpRgpu::printGlobalParameters()
{
	printf("N %d\n",N);
	printf("(Ntheta,Nrho) (%d,%d)\n",Ntheta,Nrho);
	printf("Nspan %d\n",Nspan);
	printf("Nslices %d\n",Nslices);
	printf("Nproj %d\n",Nproj);
	printf("Ntheta_R2C %d\n",Ntheta_R2C);
	printf("erho ");
	for(int i=0;i<3;i++) printf("%f ",erho[i]);
	printf("\n");
}
void lpRgpu::printFwdParameters()
{
	printf("lp2C1[0][0] %f\n",fgs->lp2C1[0][0]);
	printf("lp2C2[0][0] %f\n",fgs->lp2C2[0][0]);
	printf("p2lp1[0][0] %f\n",fgs->p2lp1[0][0]);
	printf("p2lp2[0][0] %f\n",fgs->p2lp2[0][0]);
	printf("cids[0] %d\n",fgs->cidsfwd[0]);
	printf("Ncidsfwd %d\n",fgs->Ncidsfwd);
	printf("fZ (%f,%f)\n",fZfwd[0].x,fZfwd[0].y);
	printf("\n");
}
void lpRgpu::printAdjParameters()
{
	printf("C2lp1[0][0] %f\n",ags->C2lp1[0][0]);
	printf("C2lp2[0][0] %f\n",ags->C2lp2[0][0]);
	printf("lp2p1[0][0] %f\n",ags->lp2p1[0][0]);
	printf("lp2p2[0][0] %f\n",ags->lp2p2[0][0]);
	printf("lpids[0][0] %d\n",ags->lpidsadj[0]);
	printf("Ncidsadj %d\n",ags->Ncidsadj);
	printf("Nlpidsadj %d\n",ags->Nlpidsadj);
	printf("fZadj (%f,%f)\n",fZadj[0].x,fZadj[0].y);
	printf("\n");
}

void lpRgpu::readGlobalParametersArr(float* params)
{
	N = (int)params[0];
	N0 = (int)params[1];
	Ntheta = (int)params[2];
	Nrho = (int)params[3];
	Nspan = (int)params[4];
	Nproj = (int)params[5];
	Nslices = (int)params[6];
	cor = (int)params[7];
	osangles = (int)params[8];
	interp_type = (int)params[9];
	erho = new float[Nrho*Ntheta];
	memcpy(erho,&params[10],Nrho*Ntheta*sizeof(float));
	Ntheta_R2C=(int)(Ntheta/2.0)+1;
}

void lpRgpu::readFwdParametersArr(int* paramsi, float* paramsf)
{
	int shifti=0;	
	int shiftf=0;

	fgs->Npids=new int[Nspan];
	memcpy(fgs->Npids,&paramsi[shifti],Nspan*sizeof(int));shifti+=Nspan;

	fgs->pids=new int*[Nspan];
	for(int k=0;k<Nspan;k++) 
	{
		fgs->pids[k]=new int[fgs->Npids[k]];
		memcpy(fgs->pids[k],&paramsi[shifti],fgs->Npids[k]*sizeof(int));
		shifti+=fgs->Npids[k];
	}

	fgs->lp2C1=new float*[Nspan];
	fgs->lp2C2=new float*[Nspan];
	fgs->p2lp1=new float*[Nspan];
	fgs->p2lp2=new float*[Nspan];

	fgs->Ncidsfwd = paramsi[shifti];shifti++;
	fgs->cidsfwd=new int[fgs->Ncidsfwd];

	for(int i=0;i<Nspan;i++)
	{
		fgs->lp2C1[i]=new float[fgs->Ncidsfwd];
		fgs->lp2C2[i]=new float[fgs->Ncidsfwd];
	}
	for(int i=0;i<Nspan;i++)
	{
		fgs->p2lp1[i]=new float[fgs->Npids[i]];
		fgs->p2lp2[i]=new float[fgs->Npids[i]];
	}
	for(int k=0;k<Nspan;k++) {memcpy(fgs->lp2C1[k],&paramsf[shiftf],fgs->Ncidsfwd*sizeof(float));shiftf+=fgs->Ncidsfwd;}
	for(int k=0;k<Nspan;k++) {memcpy(fgs->lp2C2[k],&paramsf[shiftf],fgs->Ncidsfwd*sizeof(float));shiftf+=fgs->Ncidsfwd;}
	for(int k=0;k<Nspan;k++) {memcpy(fgs->p2lp1[k],&paramsf[shiftf],fgs->Npids[k]*sizeof(float));shiftf+=fgs->Npids[k];}
	for(int k=0;k<Nspan;k++) {memcpy(fgs->p2lp2[k],&paramsf[shiftf],fgs->Npids[k]*sizeof(float));shiftf+=fgs->Npids[k];}
	memcpy(fgs->cidsfwd,&paramsi[shifti],fgs->Ncidsfwd*sizeof(int));shifti+=fgs->Ncidsfwd;
	fZfwd=new float2[Ntheta_R2C*Nrho];
	memcpy(fZfwd,&paramsf[shiftf],2*Ntheta_R2C*Nrho*sizeof(float));shiftf+=Ntheta_R2C*Nrho*2;

}

void lpRgpu::readAdjParametersArr(int* paramsi, float* paramsf)
{
	int shifti=0;	
	int shiftf=0;
	ags->C2lp1=new float*[Nspan];
	ags->C2lp2=new float*[Nspan];
	ags->lp2p1=new float*[Nspan];
	ags->lp2p2=new float*[Nspan];
	ags->lp2p1w=new float*[Nspan];
	ags->lp2p2w=new float*[Nspan];
	ags->Ncidsadj=paramsi[shifti];shifti++;
	ags->Nlpidsadj=paramsi[shifti];shifti++;
	ags->Nwids=paramsi[shifti];shifti++;

	for(int i=0;i<Nspan;i++)
	{
		ags->C2lp1[i]=new float[ags->Ncidsadj];
		ags->C2lp2[i]=new float[ags->Ncidsadj];
		ags->lp2p1[i]=new float[ags->Nlpidsadj];
		ags->lp2p2[i]=new float[ags->Nlpidsadj];
		ags->lp2p1w[i]=new float[ags->Nwids];
		ags->lp2p2w[i]=new float[ags->Nwids];
	}
	for(int i=0;i<Nspan;i++) {memcpy(ags->C2lp1[i],&paramsf[shiftf],ags->Ncidsadj*sizeof(float));shiftf+=ags->Ncidsadj;}
	for(int i=0;i<Nspan;i++) {memcpy(ags->C2lp2[i],&paramsf[shiftf],ags->Ncidsadj*sizeof(float));shiftf+=ags->Ncidsadj;}
	for(int i=0;i<Nspan;i++) {memcpy(ags->lp2p1[i],&paramsf[shiftf],ags->Nlpidsadj*sizeof(float));shiftf+=ags->Nlpidsadj;}
	for(int i=0;i<Nspan;i++) {memcpy(ags->lp2p2[i],&paramsf[shiftf],ags->Nlpidsadj*sizeof(float));shiftf+=ags->Nlpidsadj;}
	for(int i=0;i<Nspan;i++) {memcpy(ags->lp2p1w[i],&paramsf[shiftf],ags->Nwids*sizeof(float));shiftf+=ags->Nwids;}
	for(int i=0;i<Nspan;i++) {memcpy(ags->lp2p2w[i],&paramsf[shiftf],ags->Nwids*sizeof(float));shiftf+=ags->Nwids;}
	ags->cidsadj=new int[ags->Ncidsadj];
	memcpy(ags->cidsadj,&paramsi[shifti],ags->Ncidsadj*sizeof(int));shifti+=ags->Ncidsadj;
	ags->lpidsadj=new int[ags->Nlpidsadj];
	memcpy(ags->lpidsadj,&paramsi[shifti],ags->Nlpidsadj*sizeof(int));shifti+=ags->Nlpidsadj;
	ags->wids=new int[ags->Nwids];
	memcpy(ags->wids,&paramsi[shifti],ags->Nwids*sizeof(int));shifti+=ags->Nwids;
	fZadj=new float2[Ntheta_R2C*Nrho];
	memcpy(fZadj,&paramsf[shiftf],2*Ntheta_R2C*Nrho*sizeof(float));shiftf+=2*Ntheta_R2C*Nrho;
	int flg = paramsi[shifti];shifti++;
	if(flg==1)
	{
		filter = new float[4*N];
		memcpy(filter,&paramsf[shiftf],4*N*sizeof(float));
	}
	else filter = NULL;
}

void lpRgpu::printCurrentGPUMemory(const char* str)
{
	size_t gpufree1,gputotal;
	cudaMemGetInfo(&gpufree1,&gputotal);
	if(str!=NULL)
		printf("%s gpufree=%.0fM,gputotal=%.0fM\n",str,gpufree1/(float)(1024*1024),gputotal/(float)(1024*1024));
	else
		printf("gpufree=%.0fM,gputotal=%.0fM\n",gpufree1/(float)(1024*1024),gputotal/(float)(1024*1024));
}
