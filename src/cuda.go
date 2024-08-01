package cuda

/*
#cgo LDFLAGS: -lcuda
#include <cuda_runtime.h>
#include <stdio.h>

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    }
}
*/
import "C"
import "unsafe"

// Error handling
func checkError(err C.cudaError_t, msg string) {
	if err != C.cudaSuccess {
		panic(msg + ": " + C.GoString(C.cudaGetErrorString(err)))
	}
}

// Device management
func Init() {
	var count C.int
	err := C.cudaGetDeviceCount(&count)
	checkError(err, "Failed to get device count")
	if count == 0 {
		panic("No CUDA devices found")
	}
	err = C.cudaSetDevice(0)
	checkError(err, "Failed to set device")
}

func GetDeviceCount() int {
	var count C.int
	err := C.cudaGetDeviceCount(&count)
	checkError(err, "Failed to get device count")
	return int(count)
}

func GetDeviceProperties(device int) string {
	var props C.cudaDeviceProp
	err := C.cudaGetDeviceProperties(&props, C.int(device))
	checkError(err, "Failed to get device properties")
	return C.GoString(&props.name[0])
}

// Memory management
func AllocateMemory(size int) unsafe.Pointer {
	var d_ptr unsafe.Pointer
	err := C.cudaMalloc(&d_ptr, C.size_t(size))
	checkError(err, "Failed to allocate device memory")
	return d_ptr
}

func FreeMemory(d_ptr unsafe.Pointer) {
	err := C.cudaFree(d_ptr)
	checkError(err, "Failed to free device memory")
}

func CopyToDevice(d_ptr unsafe.Pointer, h_ptr unsafe.Pointer, size int) {
	err := C.cudaMemcpy(d_ptr, h_ptr, C.size_t(size), C.cudaMemcpyHostToDevice)
	checkError(err, "Failed to copy memory to device")
}

func CopyToHost(h_ptr unsafe.Pointer, d_ptr unsafe.Pointer, size int) {
	err := C.cudaMemcpy(h_ptr, d_ptr, C.size_t(size), C.cudaMemcpyDeviceToHost)
	checkError(err, "Failed to copy memory to host")
}

// Kernel management
func LaunchKernel(kernelName string, gridDim, blockDim int, args []unsafe.Pointer) {
	// Launch the kernel with the given arguments
}

// Stream and event management
func CreateStream() C.cudaStream_t {
	var stream C.cudaStream_t
	err := C.cudaStreamCreate(&stream)
	checkError(err, "Failed to create stream")
	return stream
}

func DestroyStream(stream C.cudaStream_t) {
	err := C.cudaStreamDestroy(stream)
	checkError(err, "Failed to destroy stream")
}

func CreateEvent() C.cudaEvent_t {
	var event C.cudaEvent_t
	err := C.cudaEventCreate(&event)
	checkError(err, "Failed to create event")
	return event
}

func DestroyEvent(event C.cudaEvent_t) {
	err := C.cudaEventDestroy(event)
	checkError(err, "Failed to destroy event")
}

func RecordEvent(event C.cudaEvent_t, stream C.cudaStream_t) {
	err := C.cudaEventRecord(event, stream)
	checkError(err, "Failed to record event")
}

func SynchronizeEvent(event C.cudaEvent_t) {
	err := C.cudaEventSynchronize(event)
	checkError(err, "Failed to synchronize event")
}
