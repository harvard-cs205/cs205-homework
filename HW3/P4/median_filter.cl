#include "median9.h"

// 3x3 median filter
__kernel void median_3x3(__global __read_only float *in_values,
                         __global __write_only float *out_values,
                         float *buffer,
                         int w, int h,
                         int buf_w, int buf_h,
                         const int halo) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if ((x < w) && (y < h)) {
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
    // Load into buffer (with 1-pixel halo).
    //
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
