

//from kernel's signature we see it expects to receive pointers to three
//memory locations. Each processor on GPU will run this code below
//first we should access thread's unique global id. 
__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    
    //initialize variables.
    float sum = 0;
    size_t local_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);
    size_t local_size = get_local_size(0);

    // thread i addw x[i], x[i + get_global_size()], up to N-1
    for (size_t j = 0; global_id + j*global_size < N ; j++) {
            sum +=  x[global_id + j*global_size];
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
    for (size_t offset = local_size/2; offset > 0 ; offset >>= 1) {
        if (local_id < offset) {    
            fast[local_id] += fast[local_id + offset];
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
    
    //initialize variables
    float sum = 0;
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    int k = ceil((float)N / global_size);

    
    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (size_t i = 0; i < k ; i++) { 
        //make sure we are in bounds relative to N and our range        
        if (global_id*k + i < N) {
            sum += x[global_id*k + i]; 
            
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
    for (size_t offset = local_size/2; offset > 0 ; offset >>= 1) {
        if (local_id < offset) {    
            fast[local_id] += fast[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
