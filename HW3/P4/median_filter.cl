#include "median9.h"

//define get clamped value to take care of edge cases
//we look at x < 0, x >= w
//we look at y < 0, y >= h
//in these cases we return the closest value in-bounds
float
get_clamped_value(__global __read_only float *in_values, int w, int h, int x, int y)
{
    if (x < 0) {
        if (y < 0) {
            return in_values[w*(y+1) + x+1];
        }
        else if (y >= h) {
            return in_values[w*(y-1) + x+1];
        }
        else {
            return in_values[w*y + x+1];
        }
    }
    else if (x >= w) {
        if (y < 0) {
            return in_values[w*(y+1) + x-1];
        }
        else if (y >= h) {
            return in_values[w*(y-1) + x-1];
        }
        else {
            return in_values[w*y + x-1];
        }
    }
    else {
        if (y < 0) {
            return in_values[w*(y+1) + x];
        }
        else if (y >= h) {
            return in_values[w*(y-1) + x];
        }
        else {
            return in_values[w*y + x];
        }
    }
}

// 3x3 median filter
__kernel void
median_3x3(
           __global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo)
{

 // Global position of output pixel relative to (0,0) (black)
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup (green)
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo (red)
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = get_clamped_value(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

    // write output
    //define the 9 points fed into median9 in terms of buff_x, buff_y, buff_w
    if ((y < h) && (x < w)) {
        float s0 = buffer[buf_w *(buf_y - 1) + (buf_x - 1)];
        float s1 = buffer[buf_w *(buf_y - 1) + (buf_x)    ];
        float s2 = buffer[buf_w *(buf_y - 1) + (buf_x + 1)];
        float s3 = buffer[buf_w *(buf_y)     + (buf_x - 1)];
        float s4 = buffer[buf_w *(buf_y)     + (buf_x)    ];
        float s5 = buffer[buf_w *(buf_y)     + (buf_x + 1)];
        float s6 = buffer[buf_w *(buf_y + 1) + (buf_x - 1)];
        float s7 = buffer[buf_w *(buf_y + 1) + (buf_x)    ];
        float s8 = buffer[buf_w *(buf_y + 1) + (buf_x + 1)];
        out_values[y * w + x] = median9(s0, s1, s2, s3, s4, s5, s6, s7, s8);
    }
}