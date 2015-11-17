// Prototype - avoids compiler warning
int get_clamped_value(__global __read_only int *labels, int w, int h, int x, int y);

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

    int old_label;
    int new_label;

    // Fetch the value from the buffer the corresponds to
    // the pixel for this thread
    old_label = buffer[buf_y * buf_w + buf_x];

    // START: NEW CODE FOR PART 2 -->

    // Only update foreground pixels
//    if (old_label < (w * h)) {
//        // Fetch grandparent   
//        buffer[buf_y * buf_w + buf_x] = labels[buffer[buf_y * buf_w + buf_x]];
//    }
    
    // <-- END: NEW CODE FOR PART 2

    // START: NEW CODE FOR PART 4 -->

    // Variables to keep track of last grandparent fetch
    int last_buf_id = -1;
    int last_buf_val = -1;

    // Use a single thread in the workgroup
    if(get_local_id(0) == 0) {
        // Only update core (non-halo values)
        for (int row = halo; row < (buf_h - halo); row++) {
            for (int col = halo; col < (buf_w - halo); col++) {
                // Check if value is equal to the last fetch performed
                if (buffer[col * buf_w + row] < (w * h)) { 
                    // Use same values if it is
                    if (buffer[col * buf_w + row] == last_buf_id) {
                        buffer[col * buf_w + row] = last_buf_val;
                    }
                    else {
                        // Update if not
                        last_buf_id = buffer[col * buf_w + row];
                        last_buf_val = labels[buffer[col * buf_w + row]];
                        buffer[col * buf_w + row] = last_buf_val;
                    }
                }
            }
        }
    }

    // <-- END: NEW CODE FOR PART 4

    // Wait for local buffer values to finish updating
    barrier(CLK_LOCAL_MEM_FENCE);

    // stay in bounds
    if ((x < w) && (y < h)) {

        new_label = old_label;

        // START: NEW CODE FOR PART 1 -->

        // Only operate on foreground pixels
        if (new_label < (w * h)) {
            // Update label as minimum of pixel's label and its 4 neighbor labels
            new_label = min(new_label, buffer[buf_y * buf_w + buf_x + 1]);
            new_label = min(new_label, buffer[buf_y * buf_w + buf_x - 1]);
            new_label = min(new_label, buffer[(buf_y + 1) * buf_w + buf_x]);
            new_label = min(new_label, buffer[(buf_y - 1) * buf_w + buf_x]);
        }

        // <-- END: NEW CODE FOR PART 1

        if (new_label != old_label) {
            
            // START: NEW CODE FOR PART 3 -->

            // Merge parent regions
            atomic_min(&labels[old_label], new_label);

            // <-- END: NEW CODE FOR PART 3

            // START: NEW CODE FOR PART 5 (TESTING ONLY) -->

            // Merge parent regions
//            labels[old_label] = min(labels[old_label], new_label);

            // <-- END: NEW CODE FOR PART 5 (TESTING ONLY)

            // indicate there was a change this iteration.
            // multiple threads might write this.
            *(changed_flag) += 1;
            labels[y * w + x] = new_label;
        }
    }
}
