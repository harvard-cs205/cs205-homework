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

    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int buf_corner_x = x - lx - halo;
    int buf_corner_y = y - ly - halo;

    int buf_x = lx + halo;
    int buf_y = ly + halo;

    int real_x;
    int real_y;

    float median;

    int idx_1D = ly * get_local_size(0) + lx;

    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    if(idx_1D < buf_w){
        for(int row = 0; row < buf_h; row++){
            real_x = buf_corner_x + idx_1D;
            if(real_x < 0) real_x = 0;
            if(real_x >= w) real_x = w - 1;
            real_y = buf_corner_y + row;
            if(real_y < 0) real_y = 0;
            if(real_y >= h) real_y = h - 1;
            buffer[row * buf_w + idx_1D] = in_values[real_x + real_y * w];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    
    median = median9(buffer[(buf_y - 1) * buf_w + buf_x - 1], buffer[(buf_y - 1) * buf_w + buf_x], buffer[(buf_y - 1) * buf_w + buf_x + 1],
                     buffer[(buf_y) * buf_w + buf_x - 1],     buffer[(buf_y) * buf_w + buf_x],     buffer[(buf_y) * buf_w + buf_x + 1],
                     buffer[(buf_y + 1) * buf_w + buf_x - 1], buffer[(buf_y + 1) * buf_w + buf_x], buffer[(buf_y + 1) * buf_w + buf_x + 1]);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if(x < w && y < h) out_values[x + y * w] = median;
}
