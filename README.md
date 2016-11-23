# cuMatlab

Build your cuda applications inside Matlab.
You don't need the parallel computing toolbox for your customized applciations.

The following examples are provided as the boilerplates.

* [cudakernels] - call a vector add cuda kernel
* [cuBLAS] - run sgemm on gpu

### Software
* Ubuntu 14.04 
* Matlab 2015b 
* Cuda Toolkit 7.5


### Installation

Build the mex function for the cuda files.
Please modify the Makefile to update the compilation environment.

For the **cudakernels** example
```sh
$ cd cudakernels && make
```
The mex function for vectoradd_cuda kernel will be generated. 
```sh
$ ls
Makefile  nvmex  nvopts.sh  vectoradd_cuda.cu  vectoradd_cuda.mexa64
```
