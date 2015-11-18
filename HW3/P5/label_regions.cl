__kernel void
initialize_labels(__global __read_only int *image,
                  __global __write_only int *labels,
                  int w, int h)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < w) && (y < h)) {
        if (image[y * w + x] > 0) {
            // set each pixel > 0 to its linear index
            labels[y * w + x] = y * w + x;
        } else {
            // out of bounds, set to maximum
            labels[y * w + x] = w * h;
        }
    }
}

int
get_clamped_value(__global __read_only int *labels,
                  int w, int h,
                  int x, int y)
{
    if ((x < 0) || (x >= w) || (y < 0) || (y >= h))
        return w * h;
    return labels[y * w + x];
}

__kernel void
propagate_labels(__global __read_write int *labels,
                 __global __write_only int *changed_flag,
                 __local int *buffer,
                 int w, int h,
                 int buf_w, int buf_h,
                 const int halo)
{

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;
    
    int old_label;
    // Will store the output value
    int new_label;
    
    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = 
                get_clamped_value(labels,
                                  w, h,
                                  buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Fetch the value from the buffer the corresponds to
    // the pixel for this thread
    old_label = buffer[buf_y * buf_w + buf_x];

    // CODE FOR PARTS 2 and 4 HERE (part 4 will replace part 2)

    int max_index = w*h; // Anything greater than this is background

    // Change core values in the buffer to grandparents. But, be careful,
    // you don't want to do this multiple times...and only update core values.
    // Do not change the index to the max index though.

    //Get the value in the buffer corresponding to your core value
    //No need for a loop, as we are always operating on core values.

    // For part 2: not using a single thread
//    int cur_buf_index = buf_y*buf_w + buf_x;
//    int parent = buffer[cur_buf_index];
//    if (parent < max_index){
//        int grandparent = labels[parent];
//        buffer[cur_buf_index] = grandparent;
//    }

    // We now redo our previous code to use a single thread. This seems like a bad idea,
    // but we'll see what happens. Part 2 with a single thread, basically.
    int LS0 = get_local_size(0);
    int LS1 = get_local_size(1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lx==0) && (ly==0)){
        // We need two for loops...yuck.
        int previous_parent = -1;
        int grandparent = -1;
        for(int cur_lx=0; cur_lx<LS0; cur_lx++){
            for(int cur_ly=0; cur_ly < LS1; cur_ly++){
                int cur_buf_x = cur_lx + halo;
                int cur_buf_y = cur_ly + halo;
                int cur_buf_index = cur_buf_y*buf_w + cur_buf_x;

                int parent = buffer[cur_buf_index];
                if (parent < max_index){
                    if (parent != previous_parent){
                        previous_parent = parent;
                        grandparent = labels[parent];
                    }
                    buffer[cur_buf_index] = grandparent;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // stay in bounds, don't update the walls.
    if ((x < w) && (y < h) && (old_label < max_index)) {
        // CODE FOR PART 1 HERE

        // Get the labels of your 4 neighboring pixels
        int top = buffer[(buf_y - 1)*buf_w + (buf_x)];
        int left = buffer[(buf_y)*buf_w + (buf_x - 1)];
        int middle = buffer[buf_y*buf_w + buf_x];
        int right = buffer[(buf_y)*buf_w + (buf_x + 1)];
        int bottom = buffer[(buf_y + 1)*buf_w + (buf_x)];

        //Create an array of the neighbors
        int all_labels[5] = {top, left, middle, right, bottom};
        int new_label = max_index;

        for(int i = 0; i < 5; i++){
            int cur_label = all_labels[i];
            if(cur_label < new_label){
                new_label = cur_label;
            }
        }

        if (new_label != old_label) {
            // indicate there was a change this iteration.
            // multiple threads might write this.
            *(changed_flag) += 1;

            //For part 3: using atomics
            atomic_min(&labels[old_label], new_label);
            atomic_min(&labels[y*w + x], labels[old_label]);

            //For part 5...do we need atomics?
            //labels[old_label] = min(labels[old_label], new_label);
            //labels[y*w + x] = min(labels[y*w+x], labels[old_label]);
        }
    }
}
