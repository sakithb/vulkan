.PHONY: all run clean

all: build/vulkan

build/vulkan: main.odin | build
	odin build . -out:build/vulkan -debug

assets/shader.frag.spv: assets/shader.frag | assets
	glslc assets/shader.frag -o assets/shader.frag.spv

assets/shader.vert.spv: assets/shader.vert | assets
	glslc assets/shader.vert -o assets/shader.vert.spv

build:
	mkdir -p build

assets:
	mkdir -p assets

run: build/vulkan assets/shader.frag.spv assets/shader.vert.spv
	./build/vulkan

clean:
	rm -rf build assets/shader.frag.spv assets/shader.vert.spv
