#include "median9.h"

// get the appropriate pixel for given point in buffer
inline float FETCH(__global float *img, int img_w, int img_h, int x, int y) {
    x = x < img_w - 1 ? x : img_w - 1;
    x = x > 0 ? x : 0;
    y = y < img_h - 1 ? y : img_h - 1;
    y = y > 0 ? y : 0;
    return img[x + img_w * y];
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
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    const int bx = lx + halo;
    const int by = ly + halo;

    const int idx_1D = ly * get_local_size(0) + lx;

    if (idx_1D < buf_h) {
        for (int row = 0; row < buf_h; row++) {
            int curr_x = buf_corner_x + idx_1D;
            int curr_y = buf_corner_y + row;
            buffer[idx_1D + buf_w * row] = \
                FETCH(in_values, w, h, curr_x, curr_y);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    float median = median9(buffer[(bx - 1) + buf_w * (by - 1)],
                           buffer[bx + buf_w * (by - 1)],
                           buffer[(bx + 1) + buf_w * (by - 1)],
                           buffer[(bx - 1) + buf_w * by],
                           buffer[bx + buf_w * by],
                           buffer[(bx + 1) + buf_w * by],
                           buffer[(bx - 1) + buf_w * (by + 1)],
                           buffer[bx + buf_w * (by + 1)],
                           buffer[(bx + 1) + buf_w * (by + 1)]);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if ((x < w) && (y < h)) {
        out_values[x + w * y] = median;
    }
}

