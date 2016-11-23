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

Run the vectoradd_cuda inside Matlab.
```sh
$ matlab -nodesktop -nojvm -nosplash
>> a = rand(1, 10)
a =
    0.8147    0.9058    0.1270    0.9134    0.6324    0.0975    0.2785    0.5469    0.9575    0.9649

a1=single(a)
a1 =
    0.8147    0.9058    0.1270    0.9134    0.6324    0.0975    0.2785    0.5469    0.9575    0.9649
    
>> b = vectoradd_cuda(a1)
b =
    1.8147    1.9058    1.1270    1.9134    1.6324    1.0975    1.2785    1.5469    1.9575    1.9649
```

References
----

- http://staff.washington.edu/dushaw/epubs/Matlab_CUDA_Tutorial_2_10.pdf
- https://github.com/lolongcovas/deep-learning-faces


**Voilà! You made it.**
