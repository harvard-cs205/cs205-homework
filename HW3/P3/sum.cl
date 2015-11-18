__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    // Global id in the x dimension and Global size
    size_t idx = get_global_id(0);
    size_t global_size = get_global_size(0);

    // Local ID and Size
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.

    for (uint i = 0; idx + i*global_size < N-1; i++) 
    {   
        // Add the values with the proper stride
        sum += x[idx+i*global_size]; 
    }

    // Save the values
    fast[local_id] = sum;

    // Synchronize the threads
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
    for (uint stride=local_size/2; stride>0 ; stride >>= 1) 
    {
        // If the local_id is smaller than the stride then continue
        if (local_id < stride)
            fast[local_id] += fast[local_id + stride];
        
        // Add barrier to ensure that all adds are complete
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}

__kernel void sum_blocked(__global float* x,
                          __global float* partial,
                          __local  float* fast,
                          long N)
{

    // Local ID and Size
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);

    // Global id in the x dimension and Global size
    size_t idx = get_global_id(0);
    size_t global_size = get_global_size(0);

    float sum = 0;
    int k = ceil(float(N) / get_global_size(0));
    int ceiling = ceil((float)N / global_size);

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (int i=idx*ceiling;  i <= (idx+1)*ceiling-1; i++) 
    { 
        // Ensure that k stays in bounds relative to N
        if (i < N)
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
    for (int stride=local_size/2; stride>0; stride >>= 1) 
    {
        if (local_id < stride)
            fast[local_id] += fast[local_id + stride];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
