#include "median9.h"

// 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo)
{
    // Note: It may be easier for you to implement median filtering
    // without using the local buffer, first, then adjust your code to
    // use such a buffer after you have that working.
    
    //get global location
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    //get local location
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    
    //get corner location of buffer
    int buf_corner_x = x - lx - halo;
    int buf_corner_y = y - ly - halo;
    
    //get location of buffer
    int buf_x = lx + halo;
    int buf_y = ly + halo;

    int idx_1D = ly * get_local_size(0) + lx;

    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.
    
    if(idx_1D < buf_w){
        // replace x_id to stay in-bounds
        int x_id = buf_corner_x + idx_1D;
        if(x_id < 0) x_id = 0;
        if(x_id >= w) x_id = w - 1;

        // replace y_id to stay in-bounds
        for(int r = 0; r < buf_h; r++){
            int y_id = buf_corner_y + r;
            if(y_id < 0) y_id = 0;
            if(y_id >= h) y_id = h-1;
            buffer[r*buf_w+idx_1D] = in_values[y_id*w+x_id];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
   
    
    
// Compute 3x3 median for each pixel in core (non-halo) pixels
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    
    float median = median9(buffer[(buf_y-1)*buf_w+buf_x-1], buffer[(buf_y-1)*buf_w+buf_x], buffer[(buf_y-1)*buf_w+buf_x+1],
                           buffer[(buf_y)*buf_w+buf_x-1], buffer[(buf_y)*buf_w+buf_x], buffer[(buf_y)*buf_w+buf_x+1],
                           buffer[(buf_y+1)*buf_w+buf_x-1], buffer[(buf_y+1)*buf_w+buf_x], buffer[(buf_y+1)*buf_w+buf_x+1]);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if(x < w && y < h) out_values[x + y * w] = median;
}
