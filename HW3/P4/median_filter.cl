#include "median9.h"

// helper function to compute 1d index from (x, y)
inline int genidx(int x, int y, int w) 
{
    return y * w + x;
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
    {
        // make sure pixels are not out of the image boundaries
        int x_idx = buf_corner_x + idx_1D;
        x_idx = max(0, min(w-1, x_idx));
        for (int row = 0; row < buf_h; row++) 
        {
            int y_idx = buf_corner_y + row;
            y_idx = max(0, min(h-1, y_idx));
            buffer[genidx(idx_1D, row, buf_w)] = in_values[genidx(x_idx, y_idx, w)];
        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if (x < w && y < h)
        out_values[genidx(x, y, w)] = median9(buffer[genidx(buf_x-1, buf_y-1, buf_h)], 
                                              buffer[genidx(buf_x,   buf_y-1, buf_h)],
                                              buffer[genidx(buf_x+1, buf_y-1, buf_h)],
                                              buffer[genidx(buf_x-1, buf_y,   buf_h)],
                                              buffer[genidx(buf_x,   buf_y,   buf_h)],
                                              buffer[genidx(buf_x+1, buf_y,   buf_h)],
                                              buffer[genidx(buf_x-1, buf_y+1, buf_h)],
                                              buffer[genidx(buf_x,   buf_y+1, buf_h)],
                                              buffer[genidx(buf_x+1, buf_y+1, buf_h)]);
}
