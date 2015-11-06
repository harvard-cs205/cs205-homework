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
    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.

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

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.

    for (int i = gid*k; (i < (gid+1)*k); i++) { // YOUR CODE HERE
        if (i < N) {
            sum += x[i]; // YOUR CODE HERE
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
