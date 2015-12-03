__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    int i, j, gID, gSize, temp, lSize, loglSize;
    
    gID = get_global_id(0);
    gSize = get_global_size(0);
    lSize = get_local_size(0);
   
    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (i = gID; i < N; i += gSize) { 
         sum = sum + x[i];
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
    loglSize = 1;
    temp  = lSize >> 1;
    while (temp > 1){
        temp = temp >> 1;
        loglSize = loglSize + 1;
    }
    for (j = 1; j <= loglSize; j++) {
        if (local_id < (lSize >> j)) { 
            fast[local_id] = fast[local_id] + fast[local_id + (lSize >> j)];
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
    int k = ceil((float)N / get_global_size(0));
    int j, gID, temp, loglSize, lSize, minS;

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    lSize = get_local_size(0);
    gID = get_global_id(0);
    if (k-1 < N - k*gID){
        minS = k;
    }
    else{
        minS = N - k*gID;
    }
    for (j = 0; j < minS; j++) { 
        sum = sum + x[k*gID + j];
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
    loglSize = 1;
    temp  = lSize >> 1;
    while (temp > 1){
        temp = temp >> 1;
        loglSize = loglSize + 1;
    }
    for (j = 1; j <= loglSize; j++) {
        if (local_id < (lSize >> j)) { 
            fast[local_id] = fast[local_id] + fast[local_id + (lSize >> j)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
