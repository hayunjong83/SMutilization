#!/bin/bash
nvcc test1D.cu -g -G -I /usr/local/cuda-10.1/samples/common/inc -o test1D
./test1D
