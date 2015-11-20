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

int min5(int x1, int x2, int x3, int x4, int x5) {
	// Returns the minimum of the five integer parameters
	int min_value = x1;
	if (min_value > x2) min_value = x2;
	if (min_value > x3) min_value = x3;
	if (min_value > x4) min_value = x4;
	if (min_value > x5) min_value = x5;
	return min_value;
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
	const int local_size = get_local_size(0);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * local_size + lx;
    
    int old_label;
    // Will store the output value
    int new_label;
    
	int row;
    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) {
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = 
                get_clamped_value(labels,
                                  w, h,
                                  buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    //barrier(CLK_LOCAL_MEM_FENCE);

    // Fetch the value from the buffer the corresponds to
    // the pixel for this thread
    old_label = buffer[buf_y * buf_w + buf_x];

    // CODE FOR PARTS 2 and 4 HERE (part 4 will replace part 2)
	/* Fetch grandparents
 	(i.e. store labels[buffer[offset]] in buffer[offset])
	*/
	// global offset, global column, global row, and buffer offset to be used to fetch grandparents
	int g_offset, g_col, g_row, b_offset;
	int last_g_offset = -1;
	int last_b_offset = -1;
	
	// thread 0 performs all reads to populate buffer
	if (idx_1D == 0) {
    	for (row = 0; row < buf_h; row++) {
			for (int idx = 0; idx < buf_w; idx++){
				// offset in buffer
				b_offset = row * buf_w + idx;
				// get global offset for grandparent
				g_offset = buffer[b_offset];
				if (last_g_offset == g_offset) {
					buffer[b_offset] = buffer[last_b_offset];
				} else {
					// calculate parameters for get_clamped_value
					// row within global labels
					g_row = g_offset / w; 
					// column within global labels
					g_col = g_offset % w; 
					// fetch grandparent
			        buffer[b_offset] = get_clamped_value(labels, w, h, g_col, g_row);
					last_g_offset = g_offset;
					last_b_offset = b_offset;
				}
			}
    	}	
	}
	// Make sure all threads reach the next part after
	// the grandparents are fetched
	barrier(CLK_LOCAL_MEM_FENCE);
    
    // stay in bounds
    if ((x < w) && (y < h)) {
        // CODE FOR PART 1 HERE
        // Set new_label as the minimum of old_label and the 4 pixels
		// adjacent to x,y
		
		// check if x,y is foreground pixel
		if (old_label < w * h)
			new_label = min5(old_label, 
						buffer[(buf_y - 1) * buf_w + buf_x],
						buffer[(buf_y + 1) * buf_w + buf_x],
						buffer[buf_y * buf_w + buf_x - 1],
						buffer[buf_y * buf_w + buf_x + 1]);

	        if (new_label < old_label) {
	            // CODE FOR PART 3 HERE
	            // indicate there was a change this iteration.
	            // multiple threads might write this.
	            *(changed_flag) += 1;
				atomic_min(&labels[y * w + x], new_label);
					if ((0 <= buf_corner_y + buf_y) && (buf_corner_y + buf_y < h) && (buf_corner_x + buf_x >= 0) && (buf_corner_x + buf_x < w)) {
						atomic_min(&labels[(buf_corner_y + buf_y) * w + buf_corner_x + buf_x], new_label);
					}
					
	        }
    }
}
