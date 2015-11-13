__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    int local_id = get_local_id(0);
    int i = get_global_id(0);
    int k = get_global_size(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (int idx = i; idx < N; idx += k) { 
        sum += x[idx];
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
    int local_size = get_local_size(0);
    for (int offset = (local_size >> 1); offset > 0; offset >>= 1) {
        fast[local_id] += fast[local_id + offset];
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}

__kernel void sum_blocked(__global float* x,
                          __global float* partial,
                          __local  float* fast,
                          long N)
{
    float sum = 0;
    int local_id = get_local_id(0);
    int i = get_global_id(0);
    int k = get_global_size(0);
    int L = ceil((float)N / k);

    // thread with global_id 0 should add 0..L-1
    // thread with global_id 1 should add L..2L-1
    // thread with global_id 2 should add 2L..3L-1
    // ...
    //     with L = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (int idx = L * i; idx < N && idx < L * (i + 1); idx++) {
        sum += x[idx];
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
    int local_size = get_local_size(0);
    for (int offset = (local_size >> 1); offset > 0; offset >>= 1) {
        fast[local_id] += fast[local_id + offset];
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
