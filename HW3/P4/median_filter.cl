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
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;

    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = in_values[y * w + x];
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...
    //
    // Should only use buffer, buf_x, buf_y.

    float pixels[9];
    for(int i = -1; i < 2; i++) {
      for (int j = -1; j < 2; j++) {
        pixels[3 * (i + 1) + (j + 1)] = buffer[(buf_y + i) * w  + (buf_x + j)];
      }
    }

    // write output
    if ((y < h) && (x < w)) // stay in bounds
        out_values[y * w + x] = median9(pixels[0], pixels[1], pixels[2], 
                                        pixels[3], pixels[4], pixels[5], 
                                        pixels[6], pixels[7], pixels[8]);
}

