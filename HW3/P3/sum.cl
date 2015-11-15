__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    int offset;
    uint j; //unsigned so that we can compare j to size_t local id
    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    size_t local_id = get_local_id(0);
    int i = get_global_id(0);
    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    // get thread id
    for (offset = i; offset < N; offset += global_size) { 
        sum += x[offset];
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
    for (j = local_size >> 1; j > 0; j >>= 1) {
        // only make sure j > local_id, so that we store the new sum in the position given by the 
        // lesser of the two indexes
        if (local_id < j) {
            fast[local_id] += fast[local_id + j];   
        }
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
    int offset;
    uint j;
    int local_size = get_local_size(0);
    int i = get_global_id(0);
    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    offset = k * i;
    while ((offset < k * (i+1)) && (offset < N)) {
        sum += x[offset];
        offset++;
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
    for (j = local_size >> 1; j > 0; j >>= 1) {
        // only make sure j > local_id, so that we store the new sum in the position given by the 
        // lesser of the two indexes
        if (local_id < j) {
            fast[local_id] += fast[local_id + j];   
        }
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
