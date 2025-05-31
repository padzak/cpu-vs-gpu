
# Prerequisites

## Lodepng - library for png decoding

Required `Lodepng` files are already included in the project directory.

Repository link - `https://github.com/lvandeve/lodepng/tree/master`

## Handwriting dataset

Download the English handwriting dataset from - `https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits?resource=download`

Unload contents of `handwriting/handwritten-english-characters-and-digits/combined_folder/train` to `NeuralNetwork/datasets/train`.

# CPU

### Compile

`g++ -O3 -std=c++17 cpu_net.cpp lodepng.cpp -o cpu_net`

### Run

`./cpu_net ../datasets/handwriting/train`

# GPU

## CUDA

### Compile

`nvcc -O3 -std=c++17 gpu_net.cu lodepng.cpp -o cuda_net`

### Run

`./cuda_net ../datasets/handwriting/train`

## HIP

### Compile

`hipcc -O3 -std=c++17 hip_net.cpp lodepng.cpp -o hip_net`

### Run

`./hip_net ../datasets/handwriting/train`