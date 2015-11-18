__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    
    // Get global id and size
    size_t i = get_global_id(0);
    size_t global_size = get_global_size(0);

    // Get local id and size
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    
    // Initialize sum
    float sum = 0;
    
    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (uint k = 0; i + k*global_size < N-1; k++) { 
        
        // Sum x[i] + x[i + stride] + x[i + 2*stride] + ...
        sum += x[i + k*global_size]; 
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
    for (uint stride=local_size/2; stride>0; stride >>= 1) {
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
    // Initialize get_x variables
    size_t local_id = get_local_id(0);
    uint local_size = get_local_size(0);
    
    // Get global ID
    uint i = get_global_id(0);
    int global_size = get_global_size(0);

    // Initialize L = ceiling(N/global_size) and sum
    int L = ceil((float)N / global_size);
    float sum = 0;

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (int k=i*L;  k <= (i+1)*L-1; k++) { 
        // Ensure that k stays in bounds relative to N
        if (k < N)
            sum += x[k];
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
    for (int stride=local_size/2; stride>0; stride >>= 1) {
        if (local_id < stride)
            fast[local_id] += fast[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
