#include "median9.h"

float globalGETPIX(__global __read_only float *im, int w, int h, int x, int y)
{
  if(x < 0)
  {
    x = 0;
  }
  if(y >= h)
  {
    y = h - 1;
  }
  if(y < 0)
  {
    y = 0;
  }
  if(x >= w)
  {
    x = w - 1;
  }
  return im[y*w+x];
}

float localGETPIX(__local float *buf, int w, int h, int x, int y)
{
  if(x < 0)
  {
    x = 0;
  }
  if(y >= h)
  {
    y = h - 1;
  }
  if(y < 0)
  {
    y = 0;
  }
  if(x >= w)
  {
    x = w - 1;
  }
  return buf[y*w+x];
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
  
    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = globalGETPIX(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

    // stay in bounds
    if ((x < w) && (y < h)) {
        out_values[y*w+x] = median9(localGETPIX(buffer, buf_w, buf_h, buf_x-1, buf_y-1), localGETPIX(buffer, buf_w, buf_h, buf_x, buf_y-1), localGETPIX(buffer, buf_w, buf_h, buf_x+1, buf_y-1),
                                localGETPIX(buffer, buf_w, buf_h, buf_x-1, buf_y), localGETPIX(buffer, buf_w, buf_h, buf_x, buf_y), localGETPIX(buffer, buf_w, buf_h, buf_x+1, buf_y),
                                localGETPIX(buffer, buf_w, buf_h, buf_x-1, buf_y+1), localGETPIX(buffer, buf_w, buf_h, buf_x, buf_y+1), localGETPIX(buffer, buf_w, buf_h, buf_x+1, buf_y+1));
    }
}


