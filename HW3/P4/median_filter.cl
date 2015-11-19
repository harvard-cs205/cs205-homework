#include "median9.h"

// version using local memory
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

    if (idx_1D < buf_w)
        for (int row = 0; row < buf_h; row++) {
            // restrict fetch only to valid values
            buffer[row * buf_w + idx_1D] = \
                in_values[max(0, min(w-1, buf_corner_x + idx_1D)) + max(0, min(h-1, buf_corner_y + row)) * w];
        }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    float median = median9(buffer[(buf_x - 1) + (buf_y - 1) * buf_w], \
                           buffer[(buf_x + 0) + (buf_y - 1) * buf_w], \
                           buffer[(buf_x + 1) + (buf_y - 1) * buf_w], \
                           buffer[(buf_x - 1) + (buf_y + 0) * buf_w], \
                           buffer[(buf_x + 0) + (buf_y + 0) * buf_w], \
                           buffer[(buf_x + 1) + (buf_y + 0) * buf_w], \
                           buffer[(buf_x - 1) + (buf_y + 1) * buf_w], \
                           buffer[(buf_x + 0) + (buf_y + 1) * buf_w], \
                           buffer[(buf_x + 1) + (buf_y + 1) * buf_w]);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if (y < h && x < w) {
        out_values[y * w + x] = median;
    }
}


// version using global mem directly
// 3x3 median filter
/*__kernel void
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

    float a0 = in_values[max(0, min(w-1, x + 0)) + max(0, min(h-1, y + 0)) * w];
    float a1 = in_values[max(0, min(w-1, x + 0)) + max(0, min(h-1, y + 1)) * w];
    float a2 = in_values[max(0, min(w-1, x + 0)) + max(0, min(h-1, y - 1)) * w];
    float a3 = in_values[max(0, min(w-1, x + 1)) + max(0, min(h-1, y + 0)) * w];
    float a4 = in_values[max(0, min(w-1, x + 1)) + max(0, min(h-1, y + 1)) * w];
    float a5 = in_values[max(0, min(w-1, x + 1)) + max(0, min(h-1, y - 1)) * w];
    float a6 = in_values[max(0, min(w-1, x - 1)) + max(0, min(h-1, y + 1)) * w];
    float a7 = in_values[max(0, min(w-1, x - 1)) + max(0, min(h-1, y + 0)) * w];
    float a8 = in_values[max(0, min(w-1, x - 1)) + max(0, min(h-1, y - 1)) * w];

    barrier(CLK_LOCAL_MEM_FENCE);
    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    float median = median9(a0, a1, a2, a3, a4, a5, a6, a7, a8);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if (y < h && x < w) {
        out_values[y * w + x] = median;
    }
}
*/