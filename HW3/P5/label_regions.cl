//######################
//#
//# Submission by Kendrick Lo (Harvard ID: 70984997) for
//# CS 205 - Computing Foundations for Computational Science (Prof. R. Jones)
//# 
//# Homework 3 - Problem 5
//#
//######################

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

    // Fetch the value from the buffer that corresponds to
    // the pixel for this thread
    old_label = buffer[buf_y * buf_w + buf_x];

    // CODE FOR PARTS 2 and 4 HERE (part 4 will replace part 2)
    // from part 2:
    // if (old_label < w * h)
    //    buffer[buf_y * buf_w + buf_x] =  \
    //        labels[buffer[buf_y * buf_w + buf_x]];

    if (get_local_id(0) == 0) {
        // assign thread with within-group index of 0 to do this job
        // for all threads in the workgroup
    
        int cached_index = w * h;   // remember most recent retrieval
                                    // initialize to dummy value
        int grandparent;
    
        for (int i=0; i<get_local_size(0); i++) {

            int parent = buffer[(buf_y * buf_w + buf_x) + i];
            
            // note: we are trying to connect foreground pixels, so
            // update buffer values for foreground pixels only
            if (parent < w * h && parent != cached_index) {
                // within bounds, but not the same 'labels' element 
                // as accessed last time -- cache this
                cached_index = parent;
                grandparent = labels[parent];
            }
            
            if (parent < w * h) {
                // use cached value to update buffer
                // (either just assigned or from recent retrieval):
                // instead of accessing same 'labels' element as last time
                // use 'cached' value instead (i.e. as stored in memory
                // private to thread with local id 0)
                buffer[(buf_y * buf_w + buf_x) + i] = grandparent; 
            }
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is updated
    barrier(CLK_LOCAL_MEM_FENCE);

    // stay in bounds
    if ((x < w) && (y < h)) {
        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.
        
        // check that the pixel is a foreground pixel
        // "The code we have provided initializes the label image, where
        //  each foreground pixel stars with its offset index within the
        //  image, and background pixels are given a value larger than 
        //  any foreground pixel."
        
        int threshold_foreground = w * h;
        new_label = old_label;

        if (old_label < threshold_foreground) {

            // in this case, it is not necessary to do the
            // comparisons because the background labels will
            // automatically be greater, and so doing pairwise minimums
            // will always pick up only the foregound pixels;
            // however, we leave them in just in case we decide
            // to modify how we distinguish between foreground
            // and background pixels

            int neighbor_1 = buffer[(buf_y+1) * buf_w + buf_x];
            if (neighbor_1 < threshold_foreground)
                new_label = min(new_label, neighbor_1);

            int neighbor_2 = buffer[(buf_y-1) * buf_w + buf_x];
            if (neighbor_2 < threshold_foreground)
                new_label = min(new_label, neighbor_2);

            int neighbor_3 = buffer[buf_y * buf_w + (buf_x+1)];
            if (neighbor_3 < threshold_foreground)
                new_label = min(new_label, neighbor_3);

            int neighbor_4 = buffer[buf_y * buf_w + (buf_x-1)];
            if (neighbor_4 < threshold_foreground)
                new_label = min(new_label, neighbor_4);
                
        }

        if (new_label != old_label) {

            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.
            *(changed_flag) += 1;

            // previously: labels[y * w + x] = new_label;
            // note from spec for atomic_min, expecting pointer operand
            atomic_min(&labels[y * w + x], new_label);
            
            // merge parents
            // note from spec for atomic_min, expecting pointer operand
            atomic_min(&labels[old_label], new_label);
        }
    }
}
