#include "median9.h"

// try to put this function here so that the compiler reads it first
float
fetch_inbound(__global  float *in_values,
           int w, int h,
           int x, int y)
{
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    // top left corner
    if((x<0)&&(y<0)) return in_values[0];
    // down left corner
    else if (((x<0)&&(y>=h))) return in_values[(h-1)*w];
    // top right corner
    else if (((x>w)&&(y<0))) return in_values[w-1];
    // down right corner
    else if (((x>w)&&(y>=h))) return in_values[w*h-1];
    // left side
    else if (x<0) return in_values[y*w];
    // top side
    else if (y<0) return in_values[x];
    // right side
    else if (x>=w) return in_values[(y+1)*w-1];
    // down side
    else if (y>=h) return in_values[(h-1)*w+x];
    // normal cas
    else return in_values[y*w+x];


}


// 3x3 median filter
__kernel void
median_3x3(__global  float *in_values,
           __global  float *out_values,
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

    if (idx_1D < buf_w)
        //should also be asking for values outside the image, fetch_inbound makes sure it does get values inbounds
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
                fetch_inbound(in_values, w, h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...
    //
    // Should only use buffer, buf_x, buf_y.

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    // write output
    if ((y < h) && (x < w)) // stay in bounds
        out_values[y * w + x] = \
            median9(buffer[(buf_y-1) * buf_w + buf_x-1],buffer[(buf_y-1) * buf_w + buf_x],buffer[(buf_y-1) * buf_w + buf_x+1], \
                buffer[buf_y * buf_w + buf_x-1],buffer[buf_y * buf_w + buf_x],buffer[buf_y * buf_w + buf_x+1], \
                buffer[(buf_y+1) * buf_w + buf_x-1],buffer[(buf_y+1) * buf_w + buf_x],buffer[(buf_y+1) * buf_w + buf_x+1]);
}
