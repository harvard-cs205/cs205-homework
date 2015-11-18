#include "median9.h"

//######################
//#
//# Submission by Kendrick Lo (Harvard ID: 70984997) for
//# CS 205 - Computing Foundations for Computational Science (Prof. R. Jones)
//# 
//# Homework 3 - Problem 4
//#
//######################

float
get_value(__global __read_only float *in_values,
                  int w, int h,
                  int x, int y)
{
    int y_coord = y;
    int x_coord = x;

    // borderline cases - choose closest pixel
    // grid is 0-indexed
    if (x<0) x_coord = 0;
    if (x>=w) x_coord = (w-1);
    if (y<0) y_coord = 0;
    if (y>=h) y_coord = (h-1); 

    return in_values[y_coord * w + x_coord];
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

    // Load the local buffer with a halo 
    if (idx_1D < buf_w) {
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = 
                get_value(in_values, w, h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...
    // Should only use buffer, buf_x, buf_y.    
    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    // Signature: 
    // inline float median9(float s0, float s1, float s2,
    //                      float s3, float s4, float s5,
    //                      float s6, float s7, float s8)

    // **************
    // *
    // * Note: All of the above code was essentially a cut & paste
    // *       job from the load_halo.cl example.
    // *       The key point to remember is here is that no loops are
    // *       required to process all pixels in the core (non-halo).
    // *       Just like each thread (with id less than buffer width)
    // *       of a workgroup contributes to loading one pixel per row
    // *       for the workgroup's buffer, one thread of a workgroup
    // *       is responsible for storing a median value for a pixel
    // *       within the core. Therefore, we only need one call to
    // *       median9 and we store the output (i.e. the final value)
    // *       directly into the global memory, as shown below.
    // *
    // ***************
    
    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    // write output

    if ((y < h) && (x < w)) // stay in bounds
        out_values[y * w + x] = median9(buffer[(buf_y-1) * buf_w + (buf_x-1)],
                                        buffer[(buf_y-1) * buf_w + buf_x],
                                        buffer[(buf_y-1) * buf_w + (buf_x+1)],
                                        buffer[buf_y * buf_w + (buf_x-1)],
                                        buffer[buf_y * buf_w + buf_x],
                                        buffer[buf_y * buf_w + (buf_x+1)],
                                        buffer[(buf_y+1) * buf_w + (buf_x-1)],
                                        buffer[(buf_y+1) * buf_w + buf_x],
                                        buffer[(buf_y+1) * buf_w + (buf_x+1)]);
}

// Notes:  buf_y-1 * buf_w will not work properly without brackets (order ops)
//   -->  (buf_y-1) * buf_w