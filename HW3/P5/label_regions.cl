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
    
    // Will store the output value
    int old_label;
    int new_label;
    
    // My Local Neighbor Variables
    int ny_back, ny_forw, nx_up, nx_down;

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

    // Calculate Pix 
    // Initialize Grandfather Label
    int prev_label=0;
    
    // Start at the First Pixel (0, 0) location in local buffer
    if (idx_1D == 0){   
        
        // Store grandparent for comparsion
        if (old_label < w*h)
            prev_label = labels[old_label];

        // Loop through the local work group minus halo 
        // ind = [y * buf_w + x]
        for (int ind=1; ind < ((buf_w-halo) * (buf_h-halo)); ind++) {
            
            // Ensure value is within image bounds
            if (buffer[ind] < w*h) {    
                // Ensure grandfather label fetch is not equal to current label
                // If true, update the current label to grandfather
                if (buffer[ind] != prev_label)
                    buffer[ind] = labels[buffer[ind]];
            }
        }
    }  

    //Ensure that all threads are done
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // stay in bounds
    if ( (x < w) && (y < h) &&  (old_label < w*h)) {
        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.
        
        // Grab Neighbor Pixels
        // y = row, x = col 
        // Referenced [row# * width + col#]
        ny_forw = buffer[(buf_y+1) * buf_w + buf_x];
        ny_back = buffer[(buf_y-1) * buf_w + buf_x];
        nx_up = buffer[buf_y * buf_w + buf_x+1];
        nx_down = buffer[buf_y * buf_w + buf_x-1];
        
        // Find the minimum pixel value from the 4 neighbors
        new_label = min(old_label, min(ny_forw, min(ny_back, min(nx_down, nx_up))));

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.

            *(changed_flag) += 1;

            // First, compare global new label with grandparent
            atomic_min(&labels[old_label], new_label);
            
            // Now write back to the non-halo portion
            // Ensure that a larger value will never be stored
            atomic_min(&labels[y * w + x], new_label);

        } 
    } 
}