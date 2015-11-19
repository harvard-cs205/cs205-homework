#include "median9.h"

float
FETCH(__global __read_only float *in_values,
      int img_w, int img_h,
      int img_x, int img_y)
{
    if (img_x < 0) img_x = 0;
    if (img_x >= img_w) img_x = img_w - 1;
    if (img_y < 0) img_y = 0;
    if (img_y >= img_h) img_y = img_h - 1;
    
    return in_values[img_y * img_w + img_x];
}

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
    
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    
    // Coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;
    
    // Coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;
    
    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;
    
    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] =
            FETCH(in_values,
                  w, h,
                  buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }
    
    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    
    // Synchronize all threads for the buffer is loaded over
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if ((y < h) && (x < w)) {
        out_values[y * w + x] =
        median9(
                buffer[(buf_y - 1) * buf_w + buf_x - 1], buffer[(buf_y - 1) * buf_w + buf_x], buffer[(buf_y - 1) * buf_w + buf_x + 1],
                buffer[buf_y * buf_w + buf_x - 1], buffer[buf_y * buf_w + buf_x], buffer[buf_y * buf_w + buf_x + 1],
                buffer[(buf_y + 1) * buf_w + buf_x - 1], buffer[(buf_y + 1) * buf_w + buf_x], buffer[(buf_y + 1) * buf_w + buf_x + 1]
             );
    }
    
}
