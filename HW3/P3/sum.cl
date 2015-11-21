__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.

    int i = get_global_id(0);
    int k = get_global_size(0);

    for (int j = 0; (i + j*k) < N; j++) { // YOUR CODE HERE
        sum = sum + x[i + j*k]; // YOUR CODE HERE
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

    int ls = get_local_size(0);
    int offset = 0;
    int limit = 0;

    // calculate log_2(local_size)
    // = to number of shifts
    while (ls > 1) {
        ls = ls >> 1;
        limit = limit + 1;
    }

    for (int j=1; j < limit; j++) { // YOUR CODE HERE
        offset = (get_local_size(0) >> j);
        if (i+offset < N)
            fast[i] = fast[i] + fast[i + offset]; // YOUR CODE HERE
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
    int k = ceil((float)N / get_global_size(0));

    int i = get_global_id(0);
    int val = 0;
    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    //
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.


    for (val = k*i; val <= k*(i+1)-1; val++) { // YOUR CODE HERE
        if (val < N)
            sum = sum + x[val]; // YOUR CODE HERE
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

    int ls = get_local_size(0);
    int offset = 0;
    int limit = 0;

    // calculate log_2(local_size)
    // = to number of shifts
    while (ls > 1) {
        ls = ls >> 1;
        limit = limit + 1;
    }

    for (int j=1; j < limit; j++) { // YOUR CODE HERE
        offset = (get_local_size(0) >> j);
        if (i + offset < N)
            fast[i] = fast[i] + fast[i + offset]; // YOUR CODE HERE
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
