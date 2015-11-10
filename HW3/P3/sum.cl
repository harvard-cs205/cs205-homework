__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t i = get_global_id(0);
    size_t local_id = get_local_id(0);
    size_t global_size = get_global_size(0);
    size_t local_size = get_local_size(0);


    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (int j=0; (i+j*global_size)<N; j++) {
        sum += x[i + j*global_size];
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // binary reduction
    // Code taken from http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    for (uint s = local_size/2; s > 0; s >>= 1) {
        if (local_id < s) {
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
    size_t i = get_global_id(0);
    size_t local_size = get_local_size(0);
    size_t g = get_global_size(0);
    int k = ceil((float) N / g);
    long int offset;

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    for (int j=0; j<k && i*k+j<N; j++) {
        sum += x[i*k+j];
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // binary reduction
    // Code taken from http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    for (uint s = local_size/2; s > 0; s >>= 1) {
        if (local_id < s) {
            fast[local_id] += fast[local_id+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}