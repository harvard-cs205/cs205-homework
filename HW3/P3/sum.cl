// sum_coalesced()
__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);
    

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    //for (;;) { // YOUR CODE HERE
    //    ; // YOUR CODE HERE
    //}
    
    // In sum coalesced(), when get global size(0) == k, the thread with get global id(0) == i should be responsible for adding the elements at {i, i + k, i + 2k, ...} up to the end of the input.
    
    for (size_t num = 0; global_id + num * global_size < N; num++) {
        sum += x[global_id + num * global_size];
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
    //for (;;) { // YOUR CODE HERE
    //    ; // YOUR CODE HERE
    //}
    
    for (size_t offset = local_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            fast[local_id] += fast[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}

// sum_blocked()

__kernel void sum_blocked(__global float* x,
                          __global float* partial,
                          __local  float* fast,
                          long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);
    int k = ceil((float)N / global_size);

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    //// YOUR CODE HERE
    //// YOUR CODE HERE
    //
    
    //In sum blocked(), when get global size(0) == k, the thread with get global id(0) == ishould be responsible for adding the elements at {L·i,L·i+1,L·i+2,...,L·(i+1) 1}, where L = dN/ke and N is the length of the vector. Be careful not to read past the end of the input vector.
    
    for (size_t i = 0; i < k; i++) {
        if (k * global_id + i < N) {
            sum += x[k * global_id + i];
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
    //// YOUR CODE HERE
    //// YOUR CODE HERE
    //
    
    for (size_t offset = local_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            fast[local_id] += fast[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
