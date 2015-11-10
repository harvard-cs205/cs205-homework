__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N) {
    // Sum
    float sum = 0;
    size_t n = (size_t)N;
    size_t i = get_global_id(0);
    size_t local_id = get_local_id(0);
    size_t local_width = get_local_size(0);
    size_t width = get_global_size(0);
    size_t height = n / width;

    for (size_t j = 0; j <= height; j++) {
        size_t index = i + width * j;
        if (index < n){
            sum += x[index];
        }
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Binary reduction

    for (size_t offset = local_width / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            fast[local_id] += fast[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Sum at the same pace for all threads
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}

__kernel void sum_blocked(__global float* x,
                          __global float* partial,
                          __local  float* fast,
                          long N) {
    float sum = 0;
    size_t local_id = get_local_id(0);
    size_t local_width = get_local_size(0);
    size_t i = get_global_id(0);
    size_t k = (size_t)ceil((float)N / (float)get_global_size(0));
    size_t n = (size_t)N;

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    //
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (size_t j = i * k; j < 2 * i * k ; j++) {
        if (j < n) {
            sum += x[j];
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
    for (size_t offset = local_width / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            fast[local_id] += fast[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Sum at the same pace for all threads
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
