__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    int i = get_global_id(0);
    int k = get_global_size(0);
    int j;
    for (j = i;j< N ; j += k) {
        sum += x[j];
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(j = get_local_size(0)/2; j > 0; j >>= 1) {
            if(local_id < j) {
              fast[local_id] += fast[local_id+j];
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

    // thread with global_id 0 should add 0..k-1

    for (int s = k*i;s < k * (i+1);a++) {
        if (a < N) sum += x[a];
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(j = get_local_size(0)/2; j > 0; j >>= 1) {
            if(local_id < j) {
              fast[local_id] += fast[local_id+j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
