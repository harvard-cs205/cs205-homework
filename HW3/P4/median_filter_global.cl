#include "median9.h"

// 3x3 median filter
__kernel void median_3x3(__global __read_only float *in_values,
                         __global __write_only float *out_values,
                         __local float *buffer,
                         int w, int h,
                         int buf_w, int buf_h,
                         const int halo) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if ((x + 1 < w) && (y + 1 < h)) {
    float pixels[9];

    for(int i = -1; i < 2; i++) {
      for (int j = -1; j < 2; j++) {
        pixels[3 * (i + 1) + (j + 1)] = in_values[(y + i) * w  + (x + j)];
      }
    }

    out_values[y * w + x] = median9(pixels[0], pixels[1], pixels[2], 
                                    pixels[3], pixels[4], pixels[5], 
                                    pixels[6], pixels[7], pixels[8]);
  }
}
