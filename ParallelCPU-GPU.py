# -*- coding: utf-8 -*-
"""
Parallel CPU and GPU Processing - Benchmark, bandwidth, bottleneck and limitation

@author: ccgoh
3/6/2015

Documentation:
http://hubonit.com/ideas/?p=671


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from multiprocessing import Process, Array
import time
import numpy

Block = 64
Size = 128
ThreadBlock = Block * Size 

#print ("Using ThreadBlock ==", ThreadBlock)

global time_taken
MAX = 10            #Set the Average of MAX == 10 
MAX_ITER = 1000     #Set the maximum loopping
#print ("Calculating %d iterations" % (MAX_ITER))

######################
# CUDA 
######################
def CUDA():
    import pycuda.tools
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath 
    from pycuda.compiler import SourceModule

# Calculate the total time taken from GPU calculation
    start_time = cuda.Event()
    stop_time = cuda.Event()

######################
# CUDA Kernel
######################
    mod = SourceModule("""
    __global__ void cuda_module(float *a1, float *values, int N)
    {
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    for(int x = 0; x < N; x++) {
    a1[idx] = ceil(a1[idx]);
    }
    values[idx] = a1[idx];
    }
    """)

    cuda_kernel = mod.get_function("cuda_module")

    # create an array from 0 to ThreadBlock
    a = numpy.linspace(0,ThreadBlock,ThreadBlock).astype(numpy.float32)

    # create an empty array
    values = numpy.zeros_like(a)

    time_taken = 0
    
    for x in range(1, MAX):
        start_time.record() # Start to record the time for CUDA calculation
        cuda_kernel(cuda.In(a), cuda.Out(values), numpy.int32(MAX_ITER), grid=(Block,1), block=(Size,1,1) )
        stop_time.record() # Stop the timer
        stop_time.synchronize()
        time_taken += start_time.time_till(stop_time)*1e-3

    total_time = time_taken / MAX
    print ("CUDA time and results:")
    print ("%.3fs, %s" % (total_time, str(values)))


######################
# CUDA Array
######################
    time_taken = 0
    
    for x in range(1, MAX):
        a = numpy.linspace(0,ThreadBlock,ThreadBlock).astype(numpy.float32)
        array_cuda = gpuarray.to_gpu(a)

        start_time.record() # Start to record the time for CUDA calculation
        for i in range(MAX_ITER):
            array_cuda = pycuda.cumath.ceil(array_cuda)
        stop_time.record() # Stop the timer
        stop_time.synchronize()
        time_taken += start_time.time_till(stop_time)*1e-3

    total_time = time_taken / MAX
    print ("CUDA Array time and results:")
    print ("%.3fs, %s" % (total_time, str(array_cuda.get())))    
    


#############
# CPU
#############
def CPU_Module(a1):
    for i in range(MAX_ITER):
        a1 = numpy.ceil(a1)
#        print a1

def Parallel_CPU():    
    time_taken = 0
    
    for x in range(1, MAX):
        a = numpy.linspace(0,ThreadBlock,ThreadBlock).astype(numpy.float32)
        array_cpu = Array('f', a)
        Parallel_CPU = Process(target=CPU_Module, args=(array_cpu,))

        start_time = time.clock()
        Parallel_CPU.start()
        Parallel_CPU.join()
        time_taken += float(time.clock() - start_time)

    total_time = time_taken / MAX
    print ("Parallel CPU time and results:")
    print ("%.3fs, %s" %(total_time, str(array_cpu[:3])))


def CPU():
    time_taken = 0
    
    for x in range(1, MAX):
        a = numpy.linspace(0,ThreadBlock,ThreadBlock).astype(numpy.float32)
        array_cpu = Array('f', a)
        start_time = time.clock()
        CPU_Module(array_cpu)
        time_taken += float(time.clock() - start_time)

    total_time = time_taken / MAX
    print ("CPU time and results:")
    print ("%.3fs, %s" %(total_time, str(array_cpu[:3])))

    
if __name__ == '__main__':
    CUDA()
    Parallel_CPU()
    CPU()        



