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

    // Buffer indices for adjacent pixels
    const int me = buf_y * buf_w + buf_x;
    const int left = me - 1;
    const int right = me + 1;
    const int top = me - buf_w;
    const int bottom = me + buf_w;

    // Will store the output value
    int new_label;
    int old_label;

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

    // Get the value of the old label
    old_label = buffer[me];

    // PART TWO - COMMENT OUT FOR ONLY PART 1 OPTIMIZATION:
    // Now update each label to its grandparents label, but only if we are in the foreground!
    if (old_label < w * h)
        buffer[me] = labels[buffer[me]];

    // END PART TWO

    // PART 4
    // We need to store the last label we looked up
    // We know that all the labels we are looking at are in the foreground, so begin our last_label
    // variable with a value corresponding to a background pixel so we wont have any conflicts when we
    // begin
    int curr_label;
    int last_label = w * h;

    // We need one thread per workgroup - since idx_1D is a unique identifier
    // for each thread inthe workgroup, we will just take the 0 thread
 /*   if (idx_1D == 0){
        // We want to loop over all elements in the core (i.e., the workgroup)
        // Thus we need to loop over all valid lx and ly values (NOT THE WHOLE BUFFER!)
        for (int curr_idx = 0; curr_idx < buf_w * buf_h; curr_idx++){

            // Look up the current label
            curr_label = buffer[curr_idx];
            
            // Only do this for the foreground pixels
            if (curr_label < w * h){
                // If the current label is different go all the way to memory
                if (curr_label != last_label)
                    buffer[curr_idx] = labels[curr_label];
                else
                    buffer[curr_idx] = last_label;

                // Now update the last label we looked up
                last_label = buffer[curr_idx];
            }

        }
     } */

    // END PART 4

    // And make sure every thread finishes this optimization before we go updating
    // all willy-nilly
    barrier(CLK_LOCAL_MEM_FENCE);

    // stay in bounds
    if ((x < w) && (y < h)) {

        // PART 1
        // If we are in the foreground, update the value
        // This should probably be in its own function, but when I did so I got an 
        // "unknown error 9999" - the 9999 was scary enough that I just did it out here
        new_label = buffer[me];
        if (new_label < w * h){
            if (buffer[top] < new_label)
                new_label = buffer[top];
            if (buffer[bottom] < new_label)
                new_label = buffer[bottom];
            if (buffer[left] < new_label)
                new_label = buffer[left];
            if (buffer[right] < new_label)
                new_label = buffer[right];
        }

        if (new_label != old_label) {
            // DEFAULT HERE
            //*(changed_flag) += 1;
            //labels[y * w + x] = new_label;

            // PART 3 BELOW - COMMENT OUT FOR ONLY PART 2 OPTIMIZATION
            atomic_min(&labels[old_label], new_label);
            *(changed_flag) += 1;
            atomic_min(&labels[y * w + x], new_label);
            // END PART 3
        }
    }
}
