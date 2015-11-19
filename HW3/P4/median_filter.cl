#include "median9.h"

// Handles boundary conditions when retrieving pixel (x, y) from 
// an image of width img_w and height img_h
float FETCH(__global float *image, int img_w, int img_h, int x, int y){
    // Declare our boundaries
    const int min_x = 0;
    const int min_y = 0;
    const int max_x = img_w - 1;
    const int max_y = img_h - 1;

    // Clamped boundary conditions - if we ever go out of bounds,
    // just take the nearest pixel
    if (x < min_x)
        x = min_x;
    if (x > max_x)
        x = max_x;
    if (y < min_y)
        y = min_y;
    if (y > max_y)
        y = max_y;

    // Now get the pixel from the image, using 1D indexing
    return image[x + y * img_w];
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

    // Load into buffer (with 1-pixel halo).
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
    int row;

    // Now load everything into the buffer contiguously by looping over rows
    // and only using some active threads
    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++){
         buffer[row * buf_w + idx_1D] = \
            FETCH(in_values, w, h,
                         buf_corner_x + idx_1D,
                         buf_corner_y + row);
         }

    // Make sure the buffer is loaded before going on
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    // Declare variables here for easier access
    // We need the 9x9 grid surrounding our current pixel in the buffer,
    // specified by (buf_x, buf_y), in terms of 1D indices
    const int tl = buf_x - 1 + (buf_y - 1) * buf_w;
    const int tm = buf_x + (buf_y - 1) * buf_w;
    const int tr = buf_x + 1 + (buf_y - 1) * buf_w;
    const int ml = buf_x - 1 + (buf_y) * buf_w;
    const int mm = buf_x + (buf_y) * buf_w;
    const int mr = buf_x + 1 + (buf_y) * buf_w;
    const int bl = buf_x - 1 + (buf_y + 1) * buf_w;
    const int bm = buf_x + (buf_y + 1) * buf_w;
    const int br = buf_x + 1 + (buf_y + 1) * buf_w;

    // Now, for all threads inside the core we need to write to the output
    // Remember that the output image is accessed globally
    // And for fast memory access, we use the buffer in our median filter
    if ((x < w) && (y < h)){
        out_values[x + y * w] = median9(buffer[tl], buffer[tm], buffer[tr],
                                        buffer[ml], buffer[mm], buffer[mr],
                                        buffer[bl], buffer[bm], buffer[br]);
    }
}


