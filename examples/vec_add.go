package main

/*
#include <cuda_runtime.h>
extern void VecAdd(float* A, float* B, float* C, int N);
*/
import "C"
import (
	"fmt"
	"go-cuda/src/cuda"
	"unsafe"
)

func main() {
	N := 1024
	size := N * 4

	cuda.Init()

	h_A := make([]float32, N)
	h_B := make([]float32, N)
	h_C := make([]float32, N)

	for i := 0; i < N; i++ {
		h_A[i] = float32(i)
		h_B[i] = float32(i * 2)
	}

	d_A := cuda.AllocateMemory(size)
	d_B := cuda.AllocateMemory(size)
	d_C := cuda.AllocateMemory(size)

	cuda.CopyToDevice(d_A, unsafe.Pointer(&h_A[0]), size)
	cuda.CopyToDevice(d_B, unsafe.Pointer(&h_B[0]), size)

	C.VecAdd((*C.float)(d_A), (*C.float)(d_B), (*C.float)(d_C), C.int(N))

	cuda.CopyToHost(unsafe.Pointer(&h_C[0]), d_C, size)

	for i := 0; i < 10; i++ {
		fmt.Printf("h_C[%d] = %f\n", i, h_C[i])
	}

	cuda.FreeMemory(d_A)
	cuda.FreeMemory(d_B)
	cuda.FreeMemory(d_C)
}
