#include "median9.h"

//this function loads data from in_values while ensuring we are reading in range
float load_data(__global __read_only float *in_values,
  int w, int h,
  int x, int y) 
{
  //x must be between 0 and w-1 (because of the halo)
  if(x < 0) {
    x = 0;
  } 
  //as the problem states, for pixels at the image boundary, we should use the closest
  //pixel in the valid region of the image
  else if (x >= w) {
    x = w - 1;
  }

  //y must be between 0 and h-1 (because of the halo)
  if(y < 0) {
    y = 0;
  } else if (y >= h) {
    y = h - 1;
  }

  return in_values[x + y*w];
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
    //code taken from load_halo.cl

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
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
                load_data(in_values, w, h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);


    //compute, and write out values if we are still in bounds
    if ((x < w) && (y < h)) {
      out_values[x + y*w] = median9(
        buffer[(buf_y-1)*buf_w + buf_x - 1],
        buffer[(buf_y-1)*buf_w + buf_x],
        buffer[(buf_y-1)*buf_w + buf_x+1],
        buffer[(buf_y)*buf_w + buf_x - 1],
        buffer[(buf_y)*buf_w + buf_x],
        buffer[(buf_y)*buf_w + buf_x+1],
        buffer[(buf_y+1)*buf_w + buf_x - 1],
        buffer[(buf_y+1)*buf_w + buf_x],
        buffer[(buf_y+1)*buf_w + buf_x+1]
      );
    }        
}
