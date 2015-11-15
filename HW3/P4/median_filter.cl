#include "median9.h"

float FETCH(__global __read_only float *in_values, int w, int h, int x, int y);

float
FETCH(__global __read_only float *in_values,
    int w, int h,
    int x, int y)
{

    // Initialize to input (global) coordinates
    int use_x = x;
    int use_y = y;

    // Correct for corner/edge cases
    use_x = max(use_x, 0);
    use_y = max(use_y, 0);
    use_x = min(use_x, w - 1);
    use_y = min(use_y, h - 1);

    // Return pixel
    return in_values[use_y * w + use_x];
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

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;

    // Load into buffer (with 1-pixel halo).
    if (idx_1D < buf_w) {
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
                FETCH(in_values, w, h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    // Check that each thread is the valid region (x < w, y < h)
    if ((y < h) && (x < w)) {
        out_values[y * w + x] = median9(buffer[(buf_y - 1) * buf_w + buf_x - 1],    buffer[(buf_y - 1) * buf_w + buf_x],    buffer[(buf_y - 1) * buf_w + buf_x + 1],
                                        buffer[buf_y * buf_w + buf_x - 1],          buffer[buf_y * buf_w + buf_x],          buffer[buf_y * buf_w + buf_x + 1],
                                        buffer[(buf_y + 1) * buf_w + buf_x - 1],    buffer[(buf_y + 1) * buf_w + buf_x],    buffer[(buf_y + 1) * buf_w + buf_x + 1]);
    }

}
