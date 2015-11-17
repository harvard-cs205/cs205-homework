__kernel void initialize_labels(__global __read_only int *image,
                                __global __write_only int *labels,
                                int w, int h) {
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

// My OpenCL (v1.0) cannot do min(), so have to define it myself
inline int lowest(int n, int m) {
    if (n < m) return n;
    else return m;
}

inline int get_clamped_value(__global __read_only int *labels,
                             int w, int h,
                             int x, int y) {
    if ((x < 0) || (x >= w) || (y < 0) || (y >= h))
        return w * h;
    return labels[y * w + x];
}

__kernel void propagate_labels(__global __read_write int *labels,
                               __global __write_only int *changed_flag,
                               __local int *buffer,
                               int w, int h,
                               int buf_w, int buf_h,
                               const int halo) {
    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lw = get_local_size(0);
    const int lh = get_local_size(1);

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

    //Make thread 0 responsible for grandparents (Part 4)
    if (idx_1D == 0) {
        int lastFetch = -1;  // Region number
        int lastIndex = -1;
        for (int row = halo; row < lw; row++) {
            for (int column = halo; column < lh; column++) {
                int index = row * buf_w + column;
                int label = buffer[index];
                if (label < w * h) {
                    if (label == lastFetch) {
                        buffer[index] = buffer[lastIndex];
                    } else {
                        lastFetch = buffer[index];
                        lastIndex = index;
                        buffer[index] = labels[label];
                    }
                }
            }
        } 
    }

    // Part 2
    // int offset = buf_y * buf_w + buf_x;
    // if (old_label < w * h)
    //     buffer[offset] = labels[buffer[offset]];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Stay in bounds
    if ((x < w) && (y < h)) {

        // Find the lowest value
        if (old_label < w * h) { 

            // Neighbors
            int a = buffer[(buf_y + 1) * buf_w + buf_x];
            int b = buffer[(buf_y - 1) * buf_w + buf_x];
            int c = buffer[buf_y * buf_w + buf_x + 1];
            int d = buffer[buf_y * buf_w + buf_x - 1];

            int i = lowest(a, b);
            int j = lowest(c, d);
            int k = lowest(i, j);
            
            if (old_label > k)
                new_label = k;
            else
                new_label = old_label;

        } else {

            new_label = old_label;

    }
        if (new_label != old_label) {

            // indicate there was a change this iteration. 
            // multiple threads might write this.
            *(changed_flag) += 1;

            atomic_min(&labels[old_label], new_label); 

            labels[y * w + x] = new_label;         

        }
    }
}
