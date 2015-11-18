__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    uint i;
    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (i = get_global_id(0); i < N; i += get_global_size(0)) 
    { // YOUR CODE HERE
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
    

    //sample code to learn about binary reduction
    /*__kernel void reduce(__global float *a,
                     __global float *r,
                     __local float *b){
    uint gid = get_global_id(0);
    uint wid = get_group_id(0);
    uint lid = get_local_id(0);
    uint gs = get_local_size(0);

    b[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = gs/2; s > 0; s >>= 1) {
        if(lid < s) {
          b[lid] += b[lid+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0) r[wid] = b[lid];*/
    uint gs = get_local_size(0);
    uint s = gs/2;
    for (s = s; s > 0; s >>= 1)
    { // YOUR CODE HERE
        
        if (local_id < s)
            {
                fast[local_id] += fast[local_id + s];
            } // YOUR CODE HERE
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
    uint k = ceil((float)N / get_global_size(0));
    uint ls = get_local_size(0);
    uint gid = get_global_id(0); 
    uint iterr;
    uint s = ls/2;

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    iterr = gid*k;
    for (iterr = iterr; (iterr < (gid + 1) * k) ; iterr++) 
    { // YOUR CODE HERE
        if (iterr < N)
            sum += x[iterr]; // YOUR CODE HERE
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
    for (s = s; s > 0; s >>= 1)
    { // YOUR CODE HERE
        
        if (local_id < s)
            {
                fast[local_id] += fast[local_id + s];
            } // YOUR CODE HERE
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}