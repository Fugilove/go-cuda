package main

import (
	"fmt"
	"go-cuda/src/cuda"
)

func main() {
	cuda.Init()

	stream := cuda.CreateStream()
	event := cuda.CreateEvent()

	// Launch some kernels (not implemented here) and use the stream and event

	cuda.RecordEvent(event, stream)
	cuda.SynchronizeEvent(event)

	fmt.Println("Event synchronized")

	cuda.DestroyEvent(event)
	cuda.DestroyStream(stream)
}
