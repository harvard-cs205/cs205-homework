#include "median9.h"

// From HW3 P5
float
get_clamped_value(__global __read_only float *labels,
                  int w, int h,
                  int x, int y)
{
    int c_x = min(w-1, max(0, x)), c_y = min(h-1, max(0, y));
    return labels[c_y * w + c_x];
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


    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    // Based on HW3 Problem 5

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

    // Load the relevant labels to a local buffer with a halo
    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] =
                get_clamped_value(in_values,
                                  w, h,
                                  buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);


    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    const int dx[3] = {-1, 0, 1}, dy[3] = {-1, 0, 1};
    int idxArr[9];

    for( int i=0; i<3; i++ ) {
        for ( int j=0; j<3; j++ ) {
            idxArr[i*3+j] = (buf_y + dy[i])*buf_w + (buf_x + dx[j]);
        }
    }

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

    //From HW3 P5
    // stay in bounds
    if ((x < w) && (y < h)) {
        out_values[y*w + x] =
                median9( buffer[ idxArr[0] ], buffer[ idxArr[1] ], buffer[ idxArr[2] ],
                        buffer[ idxArr[3] ], buffer[ idxArr[4] ], buffer[ idxArr[5] ],
                        buffer[ idxArr[6] ], buffer[ idxArr[7] ], buffer[ idxArr[8] ] );
    }

}
