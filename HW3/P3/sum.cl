__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_size = get_global_size(0);
    int global_id = get_global_id(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (int i = global_id; i < N ;i += global_size) { // the local ID should be in the boundary of the length of x
        sum += x[i];//add the current data in to the sum
    }

    fast[local_id] = sum;//store the data into the fast array
    barrier(CLK_LOCAL_MEM_FENCE);//wait for all thread to finish

    // binary reduction
    //
    // thread i should sum fast[i] and fast[i + offset] and store back
    // in fast[i], for offset = (local_size >> j) for j from 1 to
    // log_2(local_size)
    //
    // You can assume get_local_size(0) is a power of 2.
    //
    // See http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    for(int s = local_size/2; s > 0; s >>= 1) {
        if(local_id < s) {
          fast[local_id] += fast[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)  partial[get_group_id(0)] = fast[0];

}

__kernel void sum_blocked(__global float* x,
                          __global float* partial,
                          __local  float* fast,
                          long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    int k = ceil((float)N / get_global_size(0));
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (int i = k * global_id; i < N; i++) { // YOUR CODE HERE
        sum += x[i];; // YOUR CODE HERE
        if(i == k * (global_id + 1) - 1){
            break;
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
    for(int s = local_size/2; s > 0; s >>= 1) {
        if(local_id < s) {
          fast[local_id] += fast[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
