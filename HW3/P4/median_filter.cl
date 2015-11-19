#include "median9.h"

static float
get_nearest_pixel(__global __read_only float *image,
                  int w, int h,
                  int x, int y)
{
    // Set out of bounds pixels to closest pixel.
    if (x < 0) {
        x = 0;
    }
    if (x >= w) {
        x = w - 1;
    }
    if (y < 0) {
        y = 0;
    }
    if (y >= h) {
        y = h - 1;
    }
    return image[y * w + x];
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
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;

    if (idx_1D < buf_w) {
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] =
                get_nearest_pixel(in_values, w, h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    buffer[buf_y * buf_w + buf_x] = median9(
        buffer[(buf_y-1) * buf_w + buf_x - 1], buffer[(buf_y-1) * buf_w + buf_x], buffer[(buf_y-1) * buf_w + buf_x + 1],
        buffer[buf_y * buf_w + buf_x - 1], buffer[buf_y * buf_w + buf_x], buffer[buf_y * buf_w + buf_x + 1],
        buffer[(buf_y+1) * buf_w + buf_x - 1], buffer[(buf_y+1) * buf_w + buf_x], buffer[(buf_y+1) * buf_w + buf_x + 1]);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if ((y < h) && (x < w)) // stay in bounds
        out_values[y * w + x] = buffer[buf_y * buf_w + buf_x];
}
