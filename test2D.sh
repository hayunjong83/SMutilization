#!/bin/bash
nvcc test2D.cu -g -G -I /usr/local/cuda-10.1/samples/common/inc -o test2D
./test2D
