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


    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.


    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.


    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    
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

    int row;

    // Since the kernels are in order, by loading by column per kernel, 
    // we'll actually be loading by row across kernels
    if (idx_1D < buf_w) {
        for (row = 0; row < buf_h; row++) {
            int tmp_x = idx_1D;
            int tmp_y = row;
            
            if (buf_corner_x + tmp_x < 0) {
                tmp_x++;
            } else if (buf_corner_x + tmp_x >= w) {
                tmp_x--;
            }
            
            if (buf_corner_y + tmp_y < 0) {
                tmp_y++;
            } else if (buf_corner_y + tmp_y >= h) {
                tmp_y--;
            }
                
            buffer[row * buf_w + idx_1D] = in_values[((buf_corner_y + tmp_y) * w) + buf_corner_x + tmp_x];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...
    //
    // Should only use buffer, buf_x, buf_y.

    // write output
    // Stay in bounds check is necessary due to possible 
    // images with size not nicely divisible by workgroup size
    if ((y < h) && (x < w)) {// stay in bounds
        float t1 = buffer[((buf_y - 1) * buf_w) + buf_x - 1];
        float t2 = buffer[((buf_y - 1) * buf_w) + buf_x];
        float t3 = buffer[((buf_y - 1) * buf_w) + buf_x + 1];
        float t4 = buffer[(buf_y * buf_w) + buf_x - 1];
        float t5 = buffer[(buf_y * buf_w) + buf_x];
        float t6 = buffer[(buf_y * buf_w) + buf_x + 1];
        float t7 = buffer[((buf_y + 1) * buf_w) + buf_x - 1];
        float t8 = buffer[((buf_y + 1) * buf_w) + buf_x];
        float t9 = buffer[((buf_y + 1) * buf_w) + buf_x + 1]; 
        
        out_values[y * w + x] = median9(t1, t2, t3, t4, t5, t6, t7, t8, t9);
    }
}
