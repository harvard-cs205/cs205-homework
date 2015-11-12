#include "median9.h"

// 3x3 median filter
__kernel void median_3x3(__global __read_only float *in_values,
                         __global __write_only float *out_values,
                         __local float *buffer,
                         int w, int h,
                         int buf_w, int buf_h,
                         const int halo) {

  // Global position of output pixel
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  // Local position relative to (0, 0) in workgroup
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  // coordinates of our pixel in the local buffer
  const int buf_x = lx + halo;
  const int buf_y = ly + halo;

  // 1D index of thread within our work-group
  const int localIndex = ly * get_local_size(0) + lx;
  const int bufferIndex = buf_y * get_local_size(0) + buf_x;

  if ((y < h) && (x < w)) { 

    if (localIndex < buf_w)
      for (int row = 0; row < buf_h; row++)
          if (row < buf_h && (row * y) + y < h)
            buffer[row * buf_w + localIndex] = in_values[(row * y) + y * w + x];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if ((y < h) && (x < w)) {

  float pixels[9];
  if ((buf_x + 1 < w) && (buf_y + 1 < h)) {
    for(int i = -1; i < 2; i++) {
      for (int j = -1; j < 2; j++) {
        if ((buf_y + i) < buf_h && (buf_x + j) < buf_w && (buf_y + i) >= 0 && (buf_x + j) >= 0) {
          pixels[3 * (i + 1) + (j + 1)] = buffer[(buf_y + i) * w + (buf_x + j)];
        } 
      }
    }
  }

  out_values[y * w + x] = median9(pixels[0], pixels[1], pixels[2], 
                                  pixels[3], pixels[4], pixels[5], 
                                  pixels[6], pixels[7], pixels[8]);
  }
}

  // float pixel0 = buffer[(buf_y - 1) * w + (buf_x + - 1)];
  // float pixel1 = buffer[(buf_y + - 1) * w + buf_x];
  // float pixel2 = buffer[(buf_y + - 1) * w + (buf_x + 1)];
  // float pixel3 = buffer[(buf_y) * w + (buf_x - 1)];
  // float pixel4 = buffer[(buf_y) * w + (buf_x)];
  // float pixel5 = buffer[(buf_y) * w + (buf_x + 1)];
  // float pixel6 = buffer[(buf_y + 1) * w + (buf_x - 1)];
  // float pixel7 = buffer[(buf_y + 1) * w + (buf_x )];
  // float pixel8 = buffer[(buf_y + 1) * w + (buf_x + 1)];

  //  out_values[y * w + x] = median9(pixel0, pixel1, pixel2, 
  //                                  pixel3, pixel4, pixel5, 
  //                                  pixel6, pixel7, pixel8);
