#include "median9.h"

float GETPIX(__global float *, int, int, int, int);

// 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo)
{

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

    if (idx_1D < buf_w){
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = GETPIX(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

    if ((y < h) && (x < w)) // stay in bounds
        out_values[ x + w*y ] = median9(buffer[buf_x-1 + buf_w*(buf_y-1)], buffer[buf_x + buf_w*(buf_y-1)], buffer[buf_x+1 + buf_w*(buf_y-1)],
                                        buffer[buf_x-1 + buf_w*(buf_y)],   buffer[buf_x + buf_w*(buf_y)],   buffer[buf_x+1 + buf_w*(buf_y)],
                                        buffer[buf_x-1 + buf_w*(buf_y+1)], buffer[buf_x + buf_w*(buf_y+1)], buffer[buf_x+1 + buf_w*(buf_y+1)] );
    
    /*  No buffer version
    if ((y < h) && (x < w)) // stay in bounds
        out_values[ x + w*y ] = median9(GETPIX(in_values, w, h, x-1, y-1), GETPIX(in_values, w, h, x-1, y), GETPIX(in_values, w, h, x-1, y+1),
                                    GETPIX(in_values, w, h, x,   y-1), GETPIX(in_values, w, h, x,   y), GETPIX(in_values, w, h, x,   y+1),
                                    GETPIX(in_values, w, h, x+1, y-1), GETPIX(in_values, w, h, x+1, y), GETPIX(in_values, w, h, x+1, y+1));
    */

}


// get pixel
float GETPIX(__global float *in_values, int w, int h, int i, int j){
    if(i<0){
        i=0;
    }
    if(i>=w){
        i=w-1;
    }
    if(j<0){
        j=0;
    }
    if(j>=h){
        j=h-1;
    }
    return in_values[i+w*j];
}
