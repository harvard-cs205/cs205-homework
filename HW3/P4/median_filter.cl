#include "median9.h"

float FETCH(__global __read_only float *in_values, int w, int h, int x, int y)
{
    if (x < 0) x = 0;
    if (x >= w) x = w - 1;
    if (y < 0) y = 0;
    if (y >= h) y = h - 1;
    
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

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    int line, row;
    
    float middle = 0.0f;
    
    for (row = -1; row <= 1; row ++)
    {
        for (line = -1; line <= 1; line ++)
        {
             int newX = x + line;
             int newY = y + row;
             
             buffer[(buf_y + row) * buf_w + buf_x + line] = FETCH(in_values, w, h, newX, newY);    
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    middle = median9(buffer[(buf_y - 1) * buf_w + buf_x - 1],
                     buffer[(buf_y - 1) * buf_w + buf_x + 0],
                     buffer[(buf_y - 1) * buf_w + buf_x + 1],
                     buffer[(buf_y + 0) * buf_w + buf_x - 1],
                     buffer[(buf_y + 0) * buf_w + buf_x + 0],
                     buffer[(buf_y + 0) * buf_w + buf_x + 1],
                     buffer[(buf_y + 1) * buf_w + buf_x - 1],
                     buffer[(buf_y + 1) * buf_w + buf_x + 0],
                     buffer[(buf_y + 1) * buf_w + buf_x + 1]);

    // write output
    if ((y < h) && (x < w))
        out_values[y * w + x] = middle;
}
