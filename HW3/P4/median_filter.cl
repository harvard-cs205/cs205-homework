#include "median9.h"

// 3x3 median filter
__kernel void median_3x3(__global __read_only float *in_values,
                         __global __write_only float *out_values,
                         __local float *buffer,
                         int w, int h,
                         int buf_w, int buf_h,
                         const int halo){
                

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

    // We define the buffer indices and check their bounds
    if ((y < h) && (x < w)) {
      if (idx_1D  < buf_w) {
        for (int row = 0; row < buf_h; row++) {

          int new_x = buf_corner_x + idx_1D;
          int new_y = buf_corner_y + row;

          if (new_x < 0){
            new_x = 0;
          }
          else if (new_x >= w){
            new_x = w-1;
          }
          
          if (new_y < 0){
            new_y = 0;
          }
          else if (new_y >= h){
            new_y = h-1;
          }

          buffer[row * buf_w + idx_1D] = in_values[new_y * w + new_x];

        }
      }    
    }

    barrier(CLK_LOCAL_MEM_FENCE);

     if ((y < h) && (x < w)) {
        float s0 = buffer[buf_w *(buf_y - 1) + (buf_x - 1)];
        float s1 = buffer[buf_w *(buf_y - 1) + (buf_x)];
        float s2 = buffer[buf_w *(buf_y - 1) + (buf_x + 1)];
        float s3 = buffer[buf_w *(buf_y) + (buf_x - 1)];
        float s4 = buffer[buf_w *(buf_y) + (buf_x)];
        float s5 = buffer[buf_w *(buf_y) + (buf_x + 1)];
        float s6 = buffer[buf_w *(buf_y + 1) + (buf_x - 1)];
        float s7 = buffer[buf_w *(buf_y + 1) + (buf_x)];
        float s8 = buffer[buf_w *(buf_y + 1) + (buf_x + 1)];
        out_values[y * w + x] = median9(s0, s1, s2, s3, s4, s5, s6, s7, s8);
    }
}