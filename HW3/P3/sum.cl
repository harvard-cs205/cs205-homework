__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    uint local_id = get_local_id(0);
    uint gs = get_global_size(0);
    uint gls = get_local_size(0);
    uint gid = get_global_id(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.

    for (int i = gid; i < N; i += gs) { // YOUR CODE HERE
        sum += x[i]; // YOUR CODE HERE 
    }

    fast[local_id] = sum;
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
    for (int j= gls/2; j > 0 ; j >>= 1) { // YOUR CODE HERE
        if (local_id < j){
            fast[local_id] += fast[local_id + j]; // YOUR CODE HERE
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
    uint local_id = get_local_id(0);
    int k = ceil((float)N / get_global_size(0));
    uint gid = get_global_id(0);
    uint gls = get_local_size(0);

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    if (gid*k < N) {
        for (int i=gid*k; i < (gid+1)*k; i++) { // YOUR CODE HERE
            if (i >= N) {
                break;
            }
            sum += x[i]; // YOUR CODE HERE
        }
    }

    fast[local_id] = sum;
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
    for (int j= gls/2; j > 0 ; j >>= 1) { // YOUR CODE HERE
        if (local_id < j){
            fast[local_id] += fast[local_id + j]; // YOUR CODE HERE
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
