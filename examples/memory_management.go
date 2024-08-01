package main

import (
	"fmt"
	"go-cuda/src/cuda"
)

func main() {
	size := 1024 * 4

	cuda.Init()

	d_ptr := cuda.AllocateMemory(size)
	fmt.Println("Memory allocated on device")

	cuda.FreeMemory(d_ptr)
	fmt.Println("Memory freed on device")
}
