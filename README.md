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


### Examples
##### cudakernels
Build the mex function for the cuda files.

Please modify the Makefile to update the compilation environment.

```sh
$ cd cudakernels && make
```
The mex function for vectoradd_cuda kernel will be generated. 
```sh
$ ls
Makefile  nvmex  nvopts.sh  vectoradd_cuda.cu  vectoradd_cuda.mexa64
```

Run the vectoradd_cuda() inside Matlab.
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



##### cuBLAS
Test SGEMM.
You can modify the **MEXFILES**  for your kernel function in the Makefile.

Run the sgemm_cublas() inside Matlab.

```sh
$ matlab -nodesktop -nojvm -nosplash
>> A = [1 2 3 4; 5 6 7 8]
A =
     1     2     3     4
     5     6     7     8
     
>> B = [1 5; 2 6; 3 7; 4 8]
B =
     1     5
     2     6
     3     7
     4     8
     
>> C = zeros(size(A,1), size(B,2))
C =
     0     0
     0     0
>> C_out = sgemm_cublas(0,0,1,0, single(A), single(B), single(C))
C_out =
    30    70
    70   174

>> A * B
ans =
    30    70
    70   174
```


References
----

- http://staff.washington.edu/dushaw/epubs/Matlab_CUDA_Tutorial_2_10.pdf
- https://github.com/lolongcovas/deep-learning-faces


**Voilà! You made it.**
