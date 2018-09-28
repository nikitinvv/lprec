#include"config.cuh"
#include<stdio.h>
void callErr(const char* str)
{
	printf("%s\n",str);
	printf("Reset gpu\n");
	cudaDeviceReset();
}
