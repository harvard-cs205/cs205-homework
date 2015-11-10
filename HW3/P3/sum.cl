__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;

    // Why is this a size_t?
    size_t local_id = get_local_id(0);

    // This is where our thread begins summing
    uint global_id = get_global_id(0);

    // This is our how much our thread steps by when summing
    uint step = get_global_size(0);

    // This is needed for the binary reduction 
    uint ls = get_local_size(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (int i = global_id; i < N; i += step){ // YOUR CODE HERE
        sum += x[i]; // YOUR CODE HERE 
    }

    // Now load all the partial sums into a buffer
    // And make sure that the buffer is saturated before going on
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

    // Identical code as the blog post above - this is the exact same thing as is suggested in the comments above
    for (int i = ls / 2; i > 0; i >>= 1) { // YOUR CODE HERE

        // Load relevant threads into the buffer
        if (local_id < i)
            fast[local_id] += fast[local_id + i];// YOUR CODE HERE

        // Make sure all threads do their loading before going on
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

    // Size of the overall array
    // Keeping in line with the variable naming conventions in the problem
    uint k = get_global_size(0);

    // The global id of our current thread
    uint curr_thread = get_global_id(0);

    // Keeping in line with the variable names in the problem
    // This is our 'base value' that we begin at for blocking summation
    int L = ceil((float) N / k);

    // Size of our workgroup - needed for the binary reduction
    uint ls = get_local_size(0);

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (int i = L * curr_thread; i < L * (curr_thread + 1); i += 1) { // YOUR CODE HERE
        // L * (curr_thread + 1) - 1 can sometimes go out of bounds - consider for example
        // N = 9, k = 2. The second thread will try to look at element 9 in the last iteration
        // but there are only up to 8 indices (0-indexed), so we needthis check
        if (i <= N)
            sum += x[i];
    }

    // Store elements in the local buffer
    fast[local_id] = sum;

    // Make sure the local buffer is saturated before proceeding
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
    // Identical code as the blog post above - this is the exact same thing as is suggested in the comments above
    for (int i = ls / 2; i > 0; i >>= 1) { // YOUR CODE HERE

        // Load relevant threads into the buffer
        if (local_id < i)
            fast[local_id] += fast[local_id + i];// YOUR CODE HERE

        // Make sure all threads do their loading before going on
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
