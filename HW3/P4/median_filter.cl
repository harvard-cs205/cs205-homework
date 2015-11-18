#include "median9.h"

inline float FETCH(__global float *in_values, int w, int h, int xglobal, int yglobal){
    if (xglobal < 0) xglobal = 0;
    if (yglobal < 0) yglobal = 0;
    if (xglobal >= w) xglobal = w - 1;
    if (yglobal >= h) yglobal = h - 1;

    return in_values[yglobal * w + xglobal];
}

// 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer, // Remember, shared within workgroup
           int w, int h,
           int buf_w, int buf_h,
           const int halo)
{
    // Note: It may be easier for you to implement median filtering
    // without using the local buffer, first, then adjust your code to
    // use such a buffer after you have that working.

    // Global positions of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    // Local positions relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // Coordinates of the upper left corner of the buffer in image space. Includes halo.
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    //Our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1d index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;

    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    // Populates the local buffer in parallel
    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
                FETCH(in_values, w, h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    /*Remember, in your workgroup, you are given a thread to work with.
     So, you are only responsible for one pixel. You had to help initialize
     the local memory, however, or else things would not work when other threads
     tried to access it.*/

    //We need the width of the buffer

    // Get the pixels required to get median.

    float top_left = buffer[(buf_y - 1)*buf_w + (buf_x - 1)];
    float top_middle = buffer[(buf_y - 1)*buf_w + (buf_x)];
    float top_right = buffer[(buf_y - 1)*buf_w + (buf_x + 1)];
    float left = buffer[(buf_y)*buf_w + (buf_x - 1)];
    float middle = buffer[buf_y*buf_w + buf_x];
    float right = buffer[(buf_y)*buf_w + (buf_x + 1)];
    float bottom_left = buffer[(buf_y + 1)*buf_w + (buf_x - 1)];
    float bottom_middle = buffer[(buf_y + 1)*buf_w + (buf_x)];
    float bottom_right = buffer[(buf_y + 1)*buf_w + (buf_x + 1)];

    float median_result = median9(top_left, top_middle, top_right,
                                  left, middle, right,
                                  bottom_left, bottom_middle, bottom_right);


    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

    if((x < w) && (y < h)){
        out_values[y * w + x] = median_result;
    }
}