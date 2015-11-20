
//function below just initializes labels to their linear index

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

// This just takes in a particular x, y and either returns
// the MAX value for it (w*h) or the value in labels for the
// given x, y

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
    
    //This is the old code for Part 2 below
    // if (old_label < w*h){
    //     buffer[buf_y*buf_w + buf_x] = labels[buffer[buf_y*buf_w + buf_x]];
    // }


    int wg_w = get_local_size(0);
    int wg_h = get_local_size(1);

    //have only the first thread do anything
    if((lx == 0) && (ly == 0)) {
        //initialize a bunch of variables we'll need later
        int bufIDX;
        int labBuf;
        int prev;
        int prev_label;
        int row;
        int col;
        //go through every pixel in your workgroup
        for (row=0; row < wg_w; row++){
            for(col=0; col<wg_h; col++){
                //get the index in the buffer we are looking at
                //from the row,col we know in the workgroup
                bufIDX = (row+halo)*buf_w + col + halo;
                //get the label that's stored in the buffer
                labBuf = buffer[bufIDX];
                //make sure you are still in bounds
                if(labBuf < w*h){
                    //check if you've seen this Label before
                    //because then we have it in the local buffer
                    if (labBuf == prev){
                        buffer[bufIDX] = prev_label;
                    }
                    //otherwise, make the call to global memory
                    else{
                        prev = labBuf;
                        prev_label = labels[labBuf];
                        buffer[bufIDX] = prev_label;
                    }
            }

            }

        }


    }

    barrier(CLK_LOCAL_MEM_FENCE);




    int minRow;
    int minCol;
    // stay in bounds
    if ((x < w) && (y < h)) {
        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.
        
        if(old_label < w*h){
        //Make sure that you don't accidentally try to update one of the values on the bounds.
            //Grr..in C the min function can only take the min of 2 arguments. But that's fine,
            //just take the min of up down, right left.
            minRow = min(buffer[buf_y*buf_w + buf_x - 1], buffer[buf_y*buf_w + buf_x + 1]);
            minCol = min(buffer[buf_y*buf_w - buf_w + buf_x], buffer[buf_y*buf_w +buf_w + buf_x]);
            new_label = min(old_label, min(minRow, minCol)); 

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.

            //atomic_min(p, int val) where p is a pointer to the old value and val is compared.
            //value in memory address of p is updated to the min of the old and val.

            atomic_min(&labels[old_label], new_label);

            *(changed_flag) += 1;
            // labels[y * w + x] = new_label;
            atomic_min(&labels[y*w + x], new_label);
        }
    }
    }
}
