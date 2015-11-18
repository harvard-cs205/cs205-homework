__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    uint ls = get_local_size(0);
    uint lid = get_local_id(0);
    uint gs = get_global_size(0);
    uint gid = get_global_id(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (int i=gid; i<N; i+=gs) {
        sum += x[i];
    }

    fast[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // binary reduction
    //
    // thread i should sum fast[i] and fast[i + offset] and store back
    // in fast[i], for offset = (local_size >> j) for j from 1 to
    // log_2(local_size)
    //
    // You can assume get_local_size(0) is a power of 2.
    //
    // See http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    for(uint offset = ls/2; offset > 0; offset >>= 1) {
        if(lid < offset) {
          fast[lid] += fast[lid+offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial[get_group_id(0)] = fast[0];
}

__kernel void sum_blocked(__global float* x,
                          __global float* partial,
                          __local  float* fast,
                          long N)
{
    float sum = 0;
    uint ls = get_local_size(0);
    uint lid = get_local_id(0);
    uint gs = get_global_size(0);
    uint gid = get_global_id(0);
    uint L = ceil((float)N / get_global_size(0));

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (uint i=gid*L; i<(gid+1)*L && i<N; i++) {
        sum += x[i];
    }

    fast[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // binary reduction
    //
    // thread i should sum fast[i] and fast[i + offset] and store back
    // in fast[i], for offset = (local_size >> j) for j from 1 to
    // log_2(local_size)
    //
    // You can assume get_local_size(0) is a power of 2.
    //
    // See http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    for(uint offset = ls/2; offset > 0; offset >>= 1) {
        if(lid < offset) {
          fast[lid] += fast[lid+offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial[get_group_id(0)] = fast[0];
}
