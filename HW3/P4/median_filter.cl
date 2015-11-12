#include "median9.h"


inline int to1D(int w, int x, int y)
{
    return y*w + x;
}

inline int cuttail(int size, int v)
{
    if( v<0 )
        return 0;
    if( v >= size )
        return size-1;
    return v;
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
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;
    const int idx_1D = ly * get_local_size(0) + lx;
    
    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.
    if (idx_1D < buf_w)
        for ( int row = 0; row < buf_h; ++row ) {
            buffer[row * buf_w + idx_1D] = in_values[to1D(w, cuttail(w, buf_corner_x + idx_1D), cuttail(h, buf_corner_y + row))];
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    float m = median9(  buffer[to1D(buf_w, buf_x-1, buf_y-1)], buffer[to1D(buf_w, buf_x, buf_y-1)], buffer[to1D(buf_w, buf_x+1, buf_y-1)],
                        buffer[to1D(buf_w, buf_x-1, buf_y  )], buffer[to1D(buf_w, buf_x, buf_y  )], buffer[to1D(buf_w, buf_x+1, buf_y  )],
                        buffer[to1D(buf_w, buf_x-1, buf_y+1)], buffer[to1D(buf_w, buf_x, buf_y+1)], buffer[to1D(buf_w, buf_x+1, buf_y+1)]  );

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if ( y < h && x < w ) // stay in bounds
        out_values[to1D(w,x,y)] = m;
}