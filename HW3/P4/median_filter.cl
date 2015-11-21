#include "median9.h"

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
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    const int idx_1D = ly * get_local_size(0) + lx;

     int row, tx, ty;

     if (idx_1D < buf_w){
         for (row = 0; row < buf_h; row++) {
           tx = idx_1D;
           ty = row;
          if (tx+buf_corner_x < 0){
            tx++;
          } else if(tx+buf_corner_y == w) {
            tx--;
          }
          if(ty+buf_corner_y < 0) {
            ty++;
          } else if(ty+buf_corner_y == h) {
            ty--;
          }
             buffer[ty * buf_w + tx] = \
                 in_values[w*(buf_corner_y + ty)+(buf_corner_x + tx)];
         }
        }
     barrier(CLK_LOCAL_MEM_FENCE);

     if ((y < h) && (x < w)) {// stay in bounds
        out_values[y * w + x] = \
                        median9(buffer[(buf_y-1) * buf_w + buf_x-1],\
                          buffer[(buf_y-1) * buf_w + buf_x],\
                          buffer[(buf_y-1) * buf_w + buf_x+1],\
                          buffer[buf_y * buf_w + buf_x-1], \
                          buffer[buf_y * buf_w + buf_x],\
                          buffer[buf_y * buf_w + buf_x+1],\
                          buffer[(buf_y+1) * buf_w + buf_x-1],\
                          buffer[(buf_y+1) * buf_w + buf_x],\
                          buffer[(buf_y+1) * buf_w + buf_x-1]);
        }
    // Load into buffer (with 1-pixel halo).

    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.


    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.


    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
}
