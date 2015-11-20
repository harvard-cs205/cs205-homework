__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
	int i = get_global_id(0);
	int k = get_global_size(0);
	int j = 0;

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
	
    for (j=i; j<N; j=j+k) {
	    sum = sum + x[j];
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

    for (j=get_local_size(0)/2; j>0 && j>local_id; j=j/2) { 
	// Two conditions in for loop:
	//    1) last iteration when j==1 (only two values left to sum)
	//    2) stay in bounds of the fast vector and only compute the sums that are going to be used
		fast[local_id] = fast[local_id] + fast[local_id + j]; 
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
    int i=get_global_id(0);
	int j=0;
	
    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
	
	for (j=k*i; (j < k*(i+1) && j < N); j = j+1) {
	// Two conditions in the for loop:
	//     1) Stay within the block
	//     2) Stay within the input vector
	    sum = sum + x[j];
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
   
	/////// Same as above
	for (j=get_local_size(0)/2; j>0; j=j/2) { 
        if (local_id < j) {
		fast[local_id] = fast[local_id] + fast[local_id + j]; 
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}
