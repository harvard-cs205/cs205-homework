

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


//the function below ensures that the pixels do not go out of bounds of the global 
// picture

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
    // this position is naive to the fact that the workgroup is actually in the middle
    // of the buffer with the 'halo' around it
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    //neighbors of old_value
    int neighbor_1;
    int neighbor_2;
    int neighbor_3;
    int neighbor_4;
    int grandparent;
    
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
    old_label = buffer[buf_y * buf_w + buf_x]; //changed for P5.2

    //must assign a default value to this
    new_label = old_label;

    // CODE FOR PARTS 2 and 4 HERE (part 4 will replace part 2)

    // for part 4 use an if () statement to pic out one thread to load in the group members grandparents with one thread
    // need a for () loop to ensure one thread loads in all the proper grandparents
    // We want one thread to run through and load in all the proper grandparent labels to limit global reads
    if ((lx == 0) && (ly == 0)) //Choolsing the first thread in a work group
        {

        // set the appropriate grandparent value
        if (old_label < w * h)
        {
            grandparent = labels[old_label];
        }   
            // Iterate through the rows of the buffer
            for(int row = halo; row < buf_h - halo; row++)
            {
                // Iterate through the columns of the buffer
                for(int col = halo; col < buf_w - halo; col++)
                {
                    // Conditional to make sure you are not on the wall of the maze
                    if (buffer[(ly + row) * buf_w + (lx + col)] < w * h)
                    {
                        // conditional to ensure that this pixel does not already have
                        // the grandparent value
                        if (buffer[(ly + row) * buf_w + (lx + col)] != grandparent)
                        {
                            // Populate the buffer with the appropriate 
                            // grandparent value
                            buffer[(ly + row) * buf_w + (lx + col)] = labels[buffer[(ly + row) * buf_w + (lx + col)]];
                        }

                    }
                }
            }
        }

    

    // barieier here to enesure all threads are synched when loading in the buffer
    barrier(CLK_LOCAL_MEM_FENCE);


    // stay in bounds of the maze
    if ((x < w) && (y < h)) 
    {

        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.

        // must ensure that you are not on the wall of the maze
        if (old_label < w * h)
        {
            // must assign variables for all neighbors for comparison
            neighbor_1 = buffer[(buf_y + 1) * buf_w + (buf_x)]; 
            neighbor_2 = buffer[(buf_y) * buf_w + (buf_x + 1)];
            neighbor_3 = buffer[(buf_y - 1) * buf_w + (buf_x)];
            neighbor_4 = buffer[(buf_y) * buf_w + (buf_x - 1)];

            // choose the new label with a nested min() function
            new_label = min(old_label, min(min(min(neighbor_1, neighbor_2), neighbor_3), neighbor_4));
            
            // This atomic operation stores the minimum of these two values
            // with a global memory read
            if (new_label < old_label)
            {
                // changed for P5.3, Updates the Local Position
                // child regions to merge their old and new parent 
                // whenever they change to a new label
                atomic_min(&labels[old_label], new_label); 
            }

        } 


        if (new_label != old_label) 
        {
            // CODE FOR PART 3 HERE
            // indicate there was a change this iteration.
            // multiple threads might write this.
            *(changed_flag) += 1;

            // changed for P5.3, updates the Global Position
            // child regions to merge their old and new parent 
            // whenever they change to a new label
            atomic_min(&labels[y * w + x], new_label);
           
        }
    }
}


