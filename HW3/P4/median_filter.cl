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

  // coordinates of the upper left corner of the buffer in image
  // space, including halo
  const int buf_corner_x = x - lx - halo;
  const int buf_corner_y = y - ly - halo;

  // coordinates of our pixel in the local buffer
  const int buf_x = lx + halo;
  const int buf_y = ly + halo;

  // 1D index of thread within our work-group
  const int localIndex = ly * get_local_size(0) + lx;
  const int bufferIndex = buf_y * get_local_size(0) + buf_x;

  if ((y < h) && (x < w)) { 

    if (localIndex < buf_w)
      for (int row = 0; row < buf_h; row++) {
        
        int yIndex = (buf_corner_y + ly) * row;
        int xIndex = buf_corner_x + localIndex;

        int finalIndex = yIndex + xIndex;
        
        if (x < 0 && y < 0) finalIndex = 0;
        else if (x < 0 && y >= h) finalIndex = (h - 1) * w;
        else if (x >= w && y < 0) finalIndex = w - 1;
        else if (x >= w && y >= h) finalIndex = (h - 1) * w + w - 1;
        else if (x < 0) finalIndex = yIndex;
        else if (y < 0) finalIndex = xIndex;
        else if (x >= w) finalIndex = yIndex + w - 1;
        else if (y >= h) finalIndex = (h - 1) * w + xIndex;
        buffer[row * buf_w + localIndex] = in_values[finalIndex];
      }
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

