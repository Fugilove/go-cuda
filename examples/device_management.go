package main

import (
	"fmt"
	"go-cuda/src/cuda"
)

func main() {
	cuda.Init()

	count := cuda.GetDeviceCount()
	fmt.Printf("Number of CUDA devices: %d\n", count)

	for i := 0; i < count; i++ {
		name := cuda.GetDeviceProperties(i)
		fmt.Printf("Device %d: %s\n", i, name)
	}
}
