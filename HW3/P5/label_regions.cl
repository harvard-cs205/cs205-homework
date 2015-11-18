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
static int
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

    /* part 2
    // foreground check
    if (old_label < w * h) {
        //replacing each value in buffer (at index offset) with label[buffer[offset]].
        buffer[buf_y * buf_w + buf_x] = labels[old_label];
    }
    */

    // part 4
    // fetching grandparents to use a single thread in the work group
    // make sure that it remembers the last fetch it performed, avoding repeatedly reading the same global value repeatedly
    // Looked at https://piazza.com/class/icqfmi2aryn6yv?cid=524 for help

    // we let the first thread of each work group do the work, which gaurantees one thread per group
    if (lx == 0 && ly == 0){

        // initialize  the last fetch it performed
        int last_fetch_index = -1;
        int last_fetch_label = -1;

        for (uint row = 0; row < get_local_size(0); row++) {
            for (uint col = 0; col < get_local_size(1); col++){
                
                // create buffered index 
                int current_index = (row + halo) * buf_w + (col + halo);
                int current_label = buffer[current_index];
                
                // foreground check
                if (current_label < w * h){
                    // check if same as last fetch
                    // if same, just use the last fetch
                    if (current_label == last_fetch_label){
                        buffer[current_index] = last_fetch_label;
                    }
                    else {
                        // if not, update the last fetch
                        last_fetch_index = current_label;
                        last_fetch_label = labels[last_fetch_index];
                        buffer[current_index] = last_fetch_label;
                    }
                }


            }
        }
    }
    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // stay in bounds
    // We check whether we are in bounds
    if ((x < w) && (y < h) && (old_label < w * h)) {
        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.

        // for each foreground pixel, the minimum of its 4 nieghboring piexels and itself
        int north_neighbor = buffer[(buf_y - 1) * buf_w + buf_x];
        int south_neighbor = buffer[(buf_y + 1) * buf_w + buf_x];
        int east_neighbor = buffer[(buf_y) * buf_w + (buf_x - 1)];
        int west_neighbor = buffer[(buf_y) * buf_w + (buf_x + 1)];

        new_label = min(min(min(min(north_neighbor, south_neighbor),east_neighbor), west_neighbor), old_label);

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.

            //each time a pixel changes from old label to new label, update the global labels[old label] = new label.
            atomic_min(&labels[old_label], new_label);
            *(changed_flag) += 1;

            //writing back the non-halo portion at the end of the computation
            atomic_min(&labels[y * w + x], new_label);
        }
    }
}
