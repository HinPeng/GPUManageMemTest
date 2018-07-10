#include <cuda_runtime.h>

#include <iostream>
#include <math.h>

#include <helper_cuda.h>
 
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y, float *u, float *z)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i] + u[i] + z[i];
    // y[i] = x[i] + y[i];
}
 
int main(void)
{
  size_t N = static_cast<long long>(1) << 26;
  size_t mem_size = N * sizeof(float);

  #ifdef _MANAGEMEMORY
  float *x, *y, *u, *z;
  // float *x, *y;
  
  // Allocate Unified Memory -- accessible from CPU or GPU
  checkCudaErrors(cudaMallocManaged(&x, mem_size));
  checkCudaErrors(cudaMallocManaged(&y, mem_size));
  checkCudaErrors(cudaMallocManaged(&u, mem_size));
  checkCudaErrors(cudaMallocManaged(&z, mem_size));
  #else
  float *h_x, *h_y, *h_u, *h_z;
  // float *h_x, *h_y;
  h_x = (float *)malloc(mem_size);
  h_y = (float *)malloc(mem_size);
  h_u = (float *)malloc(mem_size);
  h_z = (float *)malloc(mem_size);

  float *d_x, *d_y, *d_u, *d_z;
  // float *d_x, *d_y;
  checkCudaErrors(cudaMalloc((void**) &d_x, mem_size));
  checkCudaErrors(cudaMalloc((void**) &d_y, mem_size));
  checkCudaErrors(cudaMalloc((void**) &d_u, mem_size));
  checkCudaErrors(cudaMalloc((void**) &d_z, mem_size));
  #endif
 
  // initialize x and y arrays on the host
  for (long long i = 0; i < N; i++) {
    #ifdef _MANAGEMEMORY
    x[i] = 1.0f;
    y[i] = 2.0f;
    u[i] = 3.0f;
    z[i] = 4.0f;
    #else
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
    h_u[i] = 3.0f;
    h_z[i] = 4.0f;
    #endif
  }
 
  int device = -1;
  cudaGetDevice(&device);
  #ifdef _USEPREFETCH
  checkCudaErrors(cudaMemPrefetchAsync(x, mem_size, device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(y, mem_size, device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(u, mem_size, device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(z, mem_size, device, NULL));
  #else
  #ifndef _MANAGEMEMORY
  checkCudaErrors(cudaMemcpy(d_x, h_x, mem_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, mem_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_u, h_u, mem_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_z, h_z, mem_size, cudaMemcpyHostToDevice));
  #endif
  #endif

  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start, NULL));
  #ifdef _MANAGEMEMORY
  add<<<numBlocks, blockSize>>>(N, x, y, u, z);
  // add<<<numBlocks, blockSize>>>(N, x, y);
  #else
  add<<<numBlocks, blockSize>>>(N, d_x, d_y, d_u, d_z);
  // add<<<numBlocks, blockSize>>>(N, d_x, d_y);
  #endif

  checkCudaErrors(cudaEventRecord(stop, NULL));
  // Wait for GPU to finish before accessing on host
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  printf("Time = %.3f msec\n", msecTotal);
 
  #ifndef _MANAGEMEMORY
  checkCudaErrors(cudaMemcpy(h_y, d_y, mem_size, cudaMemcpyDeviceToHost));
  #endif
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    #ifdef _MANAGEMEMORY
    maxError = fmax(maxError, fabs(y[i]-10.0f));
    #else
    maxError = fmax(maxError, fabs(h_y[i]-10.0f));
    #endif
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  #ifdef _MANAGEMEMORY
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(y));
  checkCudaErrors(cudaFree(u));
  checkCudaErrors(cudaFree(z));
  #else
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_u));
  checkCudaErrors(cudaFree(d_z));
  free(h_x);
  free(h_y);
  free(h_u);
  free(h_z);
  #endif
 
  return 0;
}