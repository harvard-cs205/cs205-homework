#include "median9.h"

// This prevents the annoying, but inconsequential, "no previous prototype" warning.
float
GETPIX(__global __read_only float *in_values,
       int w, int h,
       int i, int j);
float
GETPIX(__global __read_only float *in_values,
             int w, int h,
             int i, int j)
{
  // Stay in bounds
  if (i < 0)
      i = 0;
  else if (i >= w)
      i = w - 1;
  if (j < 0)
      j = 0;
  else if (j >= h)
      j = h - 1;

  return in_values[j * w + i];
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
  int row;

  // Global position of output pixel
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  // printf("%d  ", in_values[x, y]);

  // Local position relative to (0, 0) in workgroup
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  // coordinates of the upper left corner of the buffer in image space, including halo
  const int buf_corner_x = x - lx - halo;
  const int buf_corner_y = y - ly - halo;

  // coordinates of our pixel in the local buffer
  const int buf_x = lx + halo;
  const int buf_y = ly + halo;

  // 1D index of thread within our work-group
  const int idx_1D = ly * get_local_size(0) + lx;

  // Load into buffer
  // Taken from load_halo.cl
  if (idx_1D < buf_w) {
    for (row = 0; row < buf_h; row++) {
        buffer[row * buf_w + idx_1D] = GETPIX(in_values, w, h,
                                              buf_corner_x + idx_1D,
                                              buf_corner_y + row);
      }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Write back the result
  if ((x < w) && (y < h)) 
    out_values[y * w + x] = median9(buffer[(buf_y - 1) * buf_w + buf_x - 1],
                                    buffer[(buf_y - 1) * buf_w + buf_x],
                                    buffer[(buf_y - 1) * buf_w + buf_x + 1],
                                    buffer[buf_y * buf_w + buf_x - 1],
                                    buffer[buf_y * buf_w + buf_x],
                                    buffer[buf_y * buf_w + buf_x + 1],
                                    buffer[(buf_y + 1) * buf_w + buf_x - 1],
                                    buffer[(buf_y + 1) * buf_w + buf_x],
                                    buffer[(buf_y + 1) * buf_w + buf_x + 1]);

}