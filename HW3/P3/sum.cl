__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    float sum = 0;
    size_t local_id = get_local_id(0);
    int global_id=get_global_id(0); 
    //float local_size=get_local_size(0);
    //int lim=(int)log2(local_size);
    int i,j,offset;

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    for (i=global_id;i<N;i=i+get_global_size(0)) { // YOUR CODE HERE
        sum=sum+x[i]; 
    }
    //printf("thread[%d], sum=%f \n",global_id,sum);
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
    j=1;
    offset=get_local_size(0)>>j;
    while(offset>1) {
    //for (j = 1; j <= lim; j++) { // YOUR CODE HERE
        offset=get_local_size(0)>>j;
        //printf("global_id[%d], offset=%d\n",global_id,offset);
        fast[local_id]=fast[local_id]+fast[local_id+offset];   
        barrier(CLK_LOCAL_MEM_FENCE);
        j++;
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
    int global_id = get_global_id(0);
    //float local_size=get_local_size(0);
    int k = ceil((float)(N) / get_global_size(0));
    //int lim=(int)log2(local_size);
    int i,j,offset;


    // thread with global_id 0 should add 0..k-1
    // thread with global_id 1 should add k..2k-1
    // thread with global_id 2 should add 2k..3k-1
    // ...
    //     with k = ceil(N / get_global_size()).
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (i=global_id*k; i <= (k*(global_id+1)-1); i++) { // YOUR CODE HERE
        sum=sum+x[i]; 
    }
    
    
    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    //printf("local size=%f, lim=%d \n",local_size,lim);
    // binary reduction
    //
    // thread i should sum fast[i] and fast[i + offset] and store back
    // in fast[i], for offset = (local_size >> j) for j from 1 to
    // log_2(local_size)
    //
    // You can assume get_local_size(0) is a power of 2.
    //
    // See http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    j=1;
    offset=get_local_size(0)>>j;
    while(offset>1) {
    //for (j = 1; j <= lim; j++) { // YOUR CODE HERE
        offset=get_local_size(0)>>j;
        //printf("global_id[%d], offset=%d\n",global_id,offset);
        fast[local_id]=fast[local_id]+fast[local_id+offset];   
        barrier(CLK_LOCAL_MEM_FENCE);
        j++;
    }

    if (local_id == 0) {
        partial[get_group_id(0)] = fast[0];
    }
}
