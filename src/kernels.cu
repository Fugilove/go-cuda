extern "C" {
    __global__ void VecAdd(float* A, float* B, float* C, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            C[i] = A[i] + B[i];
        }
    }

    __global__ void VecMul(float* A, float* B, float* C, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            C[i] = A[i] * B[i];
        }
    }
}
