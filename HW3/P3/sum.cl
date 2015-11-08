/*
x = values to be summed
partial = partial sums, essentially the output of the function (sum of partials should be sum of x), one value per workgroup.
fast = fast local memory to use for communication within a workgroup, where it runs a binary reduction.
N = length of x
*/
__kernel void sum_coalesced(__global float* x, 
                            __global float* partial, 
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    uint local_id = get_local_id(0);
    const int ls = get_local_size(0);
    const int k = get_global_size(0);
    const int i = get_global_id(0);
    int j;

    for (j = 0; i+j*k<N; j++) { 
        sum += x[ i + j*k ];
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // binary reduction
    // code from http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    for(uint s = ls/2; s > 0; s >>= 1) {
        if(local_id < s) {
          fast[local_id] += fast[local_id+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}

__kernel void sum_blocked(__global float* x,
                          __global float* partial,
                          __local  float* fast,
                          long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    int ls  = get_local_size(0);
    int gid = get_global_id(0);
    int k = ceil((float)N / get_global_size(0));

    for (int i = gid*k; (i < (gid+1)*k); i++) {  // stays in-bounds relative to local range
        if (i < N) { //stays in-bounds relative to x
            sum += x[i]; 
        }
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // binary reduction
    // code from http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    for(uint s = ls/2; s > 0; s >>= 1) {
        if(local_id < s) {
          fast[local_id] += fast[local_id+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
