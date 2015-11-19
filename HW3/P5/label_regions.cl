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
    
    // Part 2:
    // if ((x < w) && (y < h) && old_label < w*h) {
    //      buffer[buf_y * buf_w + buf_x] = labels[buffer[buf_y * buf_w + buf_x]];
    // }
    // barrier(CLK_LOCAL_MEM_FENCE);

    // Part 4: 

    // Find dimensions of local work group to iterate over within a single thread.
    uint local_x_width = get_local_size(0);
    uint local_y_width = get_local_size(1);

    // // Only operates on the thread with the local index == 0 and stay in bounds
    if (idx_1D == 0 && (x < w) && (y < h)) {
        uint old_buf_value = w*h;  
        uint new_buf_value;
        uint last_lookup;
        for (uint i = 0; i < local_x_width; i++) {
            for (uint j = 0; j < local_y_width; j++) {

                // Create a buffer index for this position in the loops
                uint buf_idx = (j + halo) * buf_w + (halo + i);

                // Look up the buffer value in this position
                new_buf_value = buffer[buf_idx];
                
                // If it is not a wall:
                if (new_buf_value < w*h) { 

                    // Check if it is different from the last buffer value
                    if (new_buf_value != old_buf_value) {
                        // If so, look up the label value at this position and update the last buffer value
                        last_lookup = labels[new_buf_value];
                        old_buf_value = new_buf_value;
                    }
                    // Either way, set the beffer value to the last label lookup
                    buffer[buf_idx] = last_lookup;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // stay in bounds
    if ((x < w) && (y < h)) {
        // CODE FOR PART 1 HERE

        // If not a wall pixel:
        if (old_label < w*h) {
            // New label is equal to the min of the pixel and its neighbors.
            new_label = min(min(min(min(buffer[(buf_y-1)*buf_w + (buf_x  )], 
                                        buffer[(buf_y)  *buf_w + (buf_x-1)]), 
                                        buffer[(buf_y)  *buf_w + (buf_x  )]), 
                                        buffer[(buf_y)  *buf_w + (buf_x+1)]), 
                                        buffer[(buf_y+1)*buf_w + (buf_x  )]);
        } else { 
            // For wall pixels, do nothing:
            new_label = old_label;
        }

        // Wait for all the threads to finish before updating labels[]
        barrier(CLK_LOCAL_MEM_FENCE);

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.
            *(changed_flag) += 1;
            // Rewrite the local memory at labels[old_label] with the minimum of the two before updating
            atomic_min(&labels[old_label], labels[new_label]);
            labels[y * w + x] = labels[old_label];
            //labels[y * w + x] = new_label;
        }
    }
}
