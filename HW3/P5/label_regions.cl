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

    int current = buf_y * buf_w + buf_x; 
    // Fetch the value from the buffer the corresponds to
    // the pixel for this thread
    old_label = buffer[current];

    // CODE FOR PARTS 2 and 4 HERE (part 4 will replace part 2)

    /*
    if (old_label < w * h)
    {
        buffer[current] = labels[old_label]; // grab grandparent
    }
    */

    
    if ((lx == 0) && (ly == 0))
    {
        int prev_key = -1000;
        int prev_result;  

        for (int i = 0; i < buf_w * buf_h; i++)
        {
            int this_label = buffer[i]; 

            if (this_label >= w * h)
                continue; 

            if (prev_key == this_label)
            {
                buffer[i] = prev_result; 
            }

            else
            {
                prev_key = this_label; 
                prev_result = labels[prev_key];
            }
        }
    }
    

    
    barrier(CLK_LOCAL_MEM_FENCE);

    // stay in bounds
    if ((x < w) && (y < h)) {
        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.
        new_label = old_label;

        if (new_label < w * h)
        {
            int this = buf_y * buf_w + buf_x; 
            new_label = 
                min(buffer[(buf_y + 1) * buf_w + buf_x], 
                min(buffer[(buf_y - 1) * buf_w + buf_x], 
                min(buffer[this + 1], 
                min(buffer[this - 1], new_label
                )))); 
        }

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.

            // 
            atomic_min(&labels[old_label], new_label);

            atomic_min(&labels[y * w + x], new_label);

            *(changed_flag) += 1;
            labels[y * w + x] = new_label;
        }
    }
}
