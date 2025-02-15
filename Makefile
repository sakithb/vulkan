.aHONY: all run clean

all: build/vulkan

build/vulkan: main.odin | build
	odin build . -out:build/vulkan -debug

shaders/shader.frag.spv: shaders/shader.frag | shaders
	glslc shaders/shader.frag -o shaders/shader.frag.spv

shaders/shader.vert.spv: shaders/shader.vert | shaders
	glslc shaders/shader.vert -o shaders/shader.vert.spv

build:
	mkdir -p build

shaders:
	mkdir -p shaders

run: build/vulkan shaders/shader.frag.spv shaders/shader.vert.spv
	./build/vulkan

clean:
	rm -rf build shaders/shader.frag.spv shaders/shader.vert.spv
