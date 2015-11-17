#include "median9.h"

// Clarification: This solution is mostly referenced from HW3 Problem 5, and
//  https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl

// Note that globally out-of-bounds pixels should be replaced
// with the nearest valid pixel's value.
static float
get_clamped_value(__global __read_only float *in_values,
                  int w, int h,
                  int x, int y)
{
    x = x < 0 ? 0 : (x >= w ? w-1 : x);
    y = y < 0 ? 0 : (y >= h ? h-1 : y);
    return in_values[y * w + x];
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

    // Load into buffer (with 1-pixel halo).
    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;

    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
            get_clamped_value(in_values, w, h,
                  buf_corner_x + idx_1D,
                  buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...
    //
    // Should only use buffer, buf_x, buf_y.

    // write output: each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    if ((y < h) && (x < w)) // stay in bounds
        out_values[y * w + x] = median9(buffer[(buf_y-1) * buf_w + buf_x -1],
                                        buffer[(buf_y-1) * buf_w + buf_x],
                                        buffer[(buf_y-1) * buf_w + buf_x +1],
                                        buffer[buf_y * buf_w + buf_x -1],
                                        buffer[buf_y * buf_w + buf_x],
                                        buffer[buf_y * buf_w + buf_x +1],
                                        buffer[(buf_y+1) * buf_w + buf_x -1],
                                        buffer[(buf_y+1) * buf_w + buf_x],
                                        buffer[(buf_y+1) * buf_w + buf_x +1]
                                        );
}
