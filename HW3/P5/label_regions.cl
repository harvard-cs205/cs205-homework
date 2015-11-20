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

//DEFINE FUNCTION TO COMPUTE MIN BETWEEN TWO NUMBERS
int
min_two(int x, int y)
{
    int min;
    if (x > y) {
        min = y;
    }
    else {
        min = x;
    }
    return min;
}

//DEFINE FUNCTION TO COMPUTE MIN AMONG CENTER AND 4 ADJACENT LOCATIONS
int
get_min(int center, int left, int right, int up, int down)
{
    int min;
    min = min_two(center, left);
    min = min_two(min, right);
    min = min_two(min, up);
    min = min_two(min, down);
    return min;
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
    // PART 2
    // if (old_label < w * h) {
    //     buffer[buf_y * buf_w + buf_x] = labels[buffer[buf_y * buf_w + buf_x]];
    // }

    // PART 4: EFFICIENT GRANDPARENTS
    if (lx == 0 && ly == 0) {
        // INITIALIZE LAST LOCATION
        int last_label = buffer[0];
        int ls_row = get_local_size(0);
        int ls_col = get_local_size(1);

        for (int row = 0; row < ls_row; row++) {
            for (int col = 0; col < ls_col; col++) {
                // INITIALIZE CURRENT LOCATION AND CURRENT LABEL FOR LATER USE
                int curr_idx = (row + halo) * buf_w + (col + halo);
                int curr_label = buffer[curr_idx];
                // CHECK IF IN BOUND
                if (curr_label < w * h) {
                    // CHECK IF THE CURRENT FETCH IS THE SAME AS THE LAST ONE
                    if (curr_label==last_label) {
                        buffer[curr_idx] = last_label;
                    }
                    else {
                        buffer[curr_idx] = labels[buffer[curr_idx]];
                        last_label = buffer[curr_idx];
                    }
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // stay in bounds
    if ((x < w) && (y < h)) {
        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.

        // new_label = old_label;

        //SET UP THE LABELS AND CALCULATE THE MEAN
        if (old_label < w * h) {
            int left = buffer[buf_y * buf_w + buf_x - 1];
            int right = buffer[buf_y * buf_w + buf_x + 1];
            int up = buffer[(buf_y - 1) * buf_w + buf_x];
            int down = buffer[(buf_y + 1) * buf_w + buf_x];


        new_label = get_min(old_label, left, right, up, down);
        }
        else {
            new_label = old_label;
        }

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            atomic_min(&labels[old_label], new_label);
            // TEST FOR PART 5
            // labels[old_label] = min(labels[old_label], new_label);

            // indicate there was a change this iteration.
            // multiple threads might write this.
            *(changed_flag) += 1;
            atomic_min(&labels[y * w + x], new_label);
            // TEST FOR PART 5
            // labels[y*w + x] = min(labels[y*w+x], labels[old_label]);
        }
    }
}
