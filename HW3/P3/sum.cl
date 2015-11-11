__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N) {
    float sum = 0;
    size_t n = (size_t)N;
    size_t i = get_global_id(0);
    size_t local_id = get_local_id(0);
    size_t local_width = get_local_size(0);
    size_t width = get_global_size(0);
    size_t height = n / width;

    // Partial sum

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
    size_t L = get_global_size(0);

    // Partial sum

    for (size_t j = i * L; j < i * L + L ; j++) {
        if (j < n) {
            sum += x[j];
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
