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
//    if (old_label < w * h) {
//        // Suppose pixel 13 has a 7 in it. That means that
//        // pixel 7 and 13 are connected to each other, because that
//        // 7 value had to travel somehow to pixel 13. So we need
//        // to merge these pixel values. In addition, we know that
//        // pixel 7 will always have a smaller or equal value
//        // to pixel 13 (which has 7 in it)
//        // because pixel 7 was initialized with value 7.
//        buffer[buf_y * buf_w + buf_x] = labels[old_label];
//    }
    // The memory read above is inefficient because it may happen
    // multiple times for different threads in the same workgroup.
    // So we have just one thread do this operation for all elements
    // of the buffer and keep the last retrieved values in its register.
    if (idx_1D == buf_w) {
        int last_read_label = -1;
        int last_read_val = -1;
        int curr_old_label;
        
        // Loop through each spot in the buffer
        for (int buf_y_i = halo; buf_y_i < buf_h - 1; buf_y_i++) {
            for (int buf_x_i = halo; buf_x_i < buf_w - 1; buf_x_i++) {
                curr_old_label = buffer[buf_y_i * buf_w + buf_x_i];
                if (curr_old_label < w * h) {
                    // if this label has already been read, then don't do a global read
                    if (last_read_label == curr_old_label) {
                        buffer[buf_y_i * buf_w + buf_x_i] = last_read_val;
                    }else {
                        // Otherwise, we need to do a global read
                        // Remember that we read in this pixel.
                        last_read_label = curr_old_label;
                        last_read_val = labels[curr_old_label];
                        buffer[buf_y_i * buf_w + buf_x_i] = last_read_val;
                    }
                }
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // stay in bounds
    if ((x < w) && (y < h)) {
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.
        new_label = old_label;
        
        // Only update the non-boundaries
        if (old_label < w * h) {
            new_label = min(new_label, buffer[(buf_y + 1) * buf_w + (buf_x)]);
            new_label = min(new_label, buffer[(buf_y - 1) * buf_w + (buf_x)]);
            new_label = min(new_label, buffer[(buf_y) * buf_w + (buf_x - 1)]);
            new_label = min(new_label, buffer[(buf_y) * buf_w + (buf_x + 1)]);
        }

        if (new_label != old_label) {
            // indicate there was a change this iteration.
            // multiple threads might write this.
            
            // The fact that this pixel is connected to
            // old label (say, 7) and new label (say, 3) means that
            // pixels 3 and 7 are connected to each other.
            // So merge their values. New label is always
            // smaller than old label.
//            atomic_min(&labels[old_label], new_label);
            labels[old_label] = min(labels[old_label], new_label);
            
            *(changed_flag) += 1;
            labels[y * w + x] = new_label;
        }
    }
}
