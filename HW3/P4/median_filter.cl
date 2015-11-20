#include "median9.h"

float
find_closest(__global __read_only float *in_values,
             int w, int h,
             int x, int y)
{
    // fix out of bounds pixels to closest valid pixel
    if (x < 0) {
        x = 0;
    }
    else if (x >= w) {
        x = w - 1;
    }

    if (y < 0) {
        y = 0;
    }
    else if (y >= h) {
        y = h - 1;
    }

    return in_values[x + y * w];
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

    // constant coordinates to remember translations between local and global
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // absolute position in local buffer
    const int x_rel = get_local_id(0);
    const int y_rel = get_local_id(1);

    // remember offset coordinate for local buffer
    const int x_buf_corner = x - x_rel - halo;
    const int y_buf_corner = y - y_rel - halo;

    // local coordinate of our pixel
    const int x_buf = x_rel + halo;
    const int y_buf = y_rel + halo;

    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    const int t_idx = x_rel + y_rel * get_local_size(0);

    if (t_idx < buf_w) {
        for (int i = 0; i < buf_h; ++i) {
            buffer[i * buf_w + t_idx] = find_closest(in_values, w, h,
                                                     x_buf_corner + t_idx,
                                                     y_buf_corner + i);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

    // only calculate and write back if in valid region
    if (x < w && y < h) {
        out_values[x + y * w] = median9(buffer[(y_buf - 1) * buf_w + x_buf - 1],
                                        buffer[(y_buf - 1) * buf_w + x_buf],
                                        buffer[(y_buf - 1) * buf_w + x_buf + 1],
                                        buffer[y_buf * buf_w + x_buf - 1],
                                        buffer[y_buf * buf_w + x_buf],
                                        buffer[y_buf * buf_w + x_buf + 1],
                                        buffer[(y_buf + 1) * buf_w + x_buf - 1],
                                        buffer[(y_buf + 1) * buf_w + x_buf],
                                        buffer[(y_buf + 1) * buf_w + x_buf + 1]);
    }
}
