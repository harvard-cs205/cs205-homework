#include "median9.h"


inline float get_values(__global float *in_values, \
        int w, int h, int new_x, int new_y){
  // check everything stays in bound
  if (new_x < 0) new_x = 0;
  if (new_y < 0) new_y = 0;
  if (new_x >= w) new_x = w - 1;
  if (new_y >= h) new_y = h - 1;
  return in_values[new_y * w + new_x];
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

    // Define variables like in class

    // global position of the pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // local position of the pixel in the workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // corner coordinates of the buffer
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of the pixel in the buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // get 1-index of the pixels
    const int idx_1D = ly * get_local_size(0) + lx;

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    if (idx_1D < buf_w){
      for (int row = 0; row < buf_h; row++){
        int new_x = buf_corner_x + idx_1D;
        int new_y = buf_corner_y + row;
        // Each thread in the valid region (x < w, y < h) should write
        // back its 3x3 neighborhood median.
        buffer[row * buf_w + idx_1D] = \
            get_values(in_values, w, h, new_x, new_y);
      }
    }
        
    //# Make sure all threads reach the next part after the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    if((x < w) && (y < h)){
      out_values[y * w + x] =\
        median9(buffer[(buf_y-1) * buf_w + buf_x -1],\
        buffer[(buf_y-1) * buf_w + buf_x],\
        buffer[(buf_y-1) * buf_w + buf_x +1],\
        buffer[buf_y * buf_w + buf_x -1],  \ 
        buffer[buf_y * buf_w + buf_x], \    
        buffer[buf_y * buf_w + buf_x +1],\
        buffer[(buf_y+1) * buf_w + buf_x -1],\
        buffer[(buf_y+1) * buf_w + buf_x],\
        buffer[(buf_y+1) * buf_w + buf_x +1]);

    }


}
