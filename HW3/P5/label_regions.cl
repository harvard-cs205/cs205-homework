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

//get clamped values addresses our edge-cases/out of bounds scenarios like in P4
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
    
    // Get local size
    int LS0 = get_local_size(0);
    int LS1 = get_local_size(1);


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

    // CODE FOR PART 2 BELOW
    //if (old_label < w*h)  {  
       //buffer[buf_y* buf_w + buf_x] = labels[buffer[buf_y* buf_w + buf_x]];
    //}
    // CODE FOR PART 4 BELOW
    
    //make sure only one thread does this optimization
    if ((lx == 0) & (ly==0)) {
        int last_pull = -1;
        int last_check = -1;
        //printf (lx);
        for (int i= halo; i < LS0 ; i++) {
            for (int j = halo; j < LS1; j++) {
                if (buffer[(j+halo)*buf_w + (i+halo)] < w*h) {
                
                    //if it just made this check, then assign it the value of last pull
                    if (buffer[(j+halo)*buf_w + (i+halo)] == last_check) {
                        buffer[(j+halo)*buf_w + (i+halo)] = last_pull;
                }
                    //otherwise go to memory and grab label value
                    else {
                        buffer[(j+halo)*buf_w + (i+halo)] = labels[buffer[(j+halo)*buf_w + (i+halo)]];
                        last_pull=buffer[(j+halo)*buf_w + (i+halo)];
                }
             }
         }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // stay in bounds
    if ((x < w) && (y < h) && (old_label < w*h)) {
        //Like in P4, we grab pixels around the pixel we care about        
        float s1 = buffer[buf_w *(buf_y - 1) + (buf_x)    ];
        float s3 = buffer[buf_w *(buf_y)     + (buf_x - 1)];
        float s5 = buffer[buf_w *(buf_y)     + (buf_x + 1)];
        float s7 = buffer[buf_w *(buf_y + 1) + (buf_x)    ];
        float array[] = {s1, s3, old_label, s5, s7}  ;      
        float minimum = array[0];        
        //find minimum value        
        for (int i = 1 ; i < 5 ; i++ ) {
            if (array[i] < minimum ) {
               minimum = array[i];
            }
        }
        new_label = minimum;

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.
            atomic_min(&labels[old_label], new_label);           
            *(changed_flag) += 1;
            labels[y * w + x] = new_label;
        }
    }
}
