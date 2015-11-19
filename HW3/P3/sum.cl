__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    long local_size = get_local_size(0); 
    long global_size = get_global_size(0); 


    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (int i = global_id; i < N; i+=global_size ) { // YOUR CODE HERE
        sum += x[i];
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
    for (uint off = local_size/2; off > 0; off >>= 1) { // YOUR CODE HERE
        if (local_id < off)
        {
            fast[local_id] += fast[local_id + off];
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
    size_t global_id = get_global_id(0);
    long local_size = get_local_size(0); 
    long global_size = get_global_size(0); 

    int k = ceil((float)N / get_global_size(0));
    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    int ix = 0; 
    for (uint i = 0; i < k; i++) { // YOUR CODE HERE
        ix = global_id * k + i;
        if (ix < N)
            sum += x[ix];
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

    for (uint off = local_size/2; off > 0; off >>= 1) { // YOUR CODE HERE
        if (local_id < off)
        {
            fast[local_id] += fast[local_id + off];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
