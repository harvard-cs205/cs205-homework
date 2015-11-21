#include "median9.h"

// prototype
float get_next(__global __read_only float *in_values, int idx_x, int idx_y, int w, int h);

// 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo)
{
  // local coordinates
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);

  // global coordinates
  int global_x = get_global_id(0);
  int global_y = get_global_id(1);

  // corrdinates of the upper left corner of the buffer square 
  int buffer_upperleft_corner_x = global_x - local_x - halo;
  int buffer_upperleft_corner_y = global_y - local_y - halo;

  // coordinates in the buffer
  int buffer_x = local_x + halo;
  int buffer_y = local_y + halo;

  // buffer index
  int local_size = get_local_size(0);
  int buffer_1D_index = local_x + (local_y * local_size);

  // load into buffer
  if (buffer_1D_index < buf_w)
    for (int line = 0; line < buf_h; line++)
      {
      buffer[buffer_1D_index + (line * buf_w)] = get_closest(in_values, buffer_upperleft_corner_x + buffer_1D_index, buffer_upperleft_corner_y + line, w, h);
      }
  barrier(CLK_LOCAL_MEM_FENCE);

  // new buffer values
  float buffer_1 = buffer[(buffer_x - 1) + buf_w * (buffer_y - 1)];
  float buffer_2 = buffer[buffer_x + (buf_w * (buffer_y - 1))];
  float buffer_3 = buffer[(buffer_x + 1) + (buf_w * (buffer_y - 1))];
  float buffer_4 = buffer[(buffer_x - 1) + (buf_w * buffer_y)];
  float buffer_5 = buffer[buffer_x + (buf_w * buffer_y)];
  float buffer_6 = buffer[(buffer_x + 1) + (buf_w * buffer_y)];
  float buffer_7 = buffer[(buffer_x - 1) + (buf_w * (buffer_y + 1))];
  float buffer_8 = buffer[buffer_x + (buf_w * (buffer_y + 1))];
  float buffer_9 = buffer[(buffer_x + 1) + buf_w * (buffer_y + 1)];                                  

  // output
  if ((global_x < w) && (global_y < h))
    out_values[global_x + (global_y * w)] = median9(buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, buffer_6, buffer_7, buffer_8, buffer_9);

}

// if value out of bounds return nearest inbound value
float get_closest(__global __read_only float *in_values, int idx_x, int idx_y, int w, int h)
{
  if (idx_x <= 0)
    idx_x = 0;
  if (idx_x >= w)
    idx_x = w - 1;
  if (idx_y <= 0)
    idx_y = 0;
  if (idx_y >= h)
    idx_y = h - 1;

  return in_values[(idx_y * w) + idx_x];
}
