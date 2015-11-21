#include "median9.h"

 
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
    int gID, lID, x, y, lx, ly, gSizeX, gSizeY, 
        lSizeX, lSizeY, xTemp, yTemp, xUse, yUse,
        buf_corner_x, buf_corner_y, buf_x, buf_y, row;

    x = get_global_id(0);
    y = get_global_id(1);
    lx = get_local_id(0);
    ly = get_local_id(1);
    gSizeX = get_global_size(0);
    gSizeY = get_global_size(1);
    lSizeX = get_local_size(0);
    lSizeY = get_local_size(1);
    
    
    gID = gSizeX*y + x;
    lID = lSizeX*ly + lx;

    buf_corner_x = x - lx - halo;
    buf_corner_y = y - ly - halo;

    buf_x = lx + halo;
    buf_y = ly + halo;

    if ((y < h) && (x < w)){
        if (lID < buf_w){
            xTemp = buf_corner_x + lID;
            xUse = xTemp;
            if (xTemp < 0){
                    xUse += 1;
                }
            if (xTemp > w - 1){
                xUse -= 1;
            }
            for (row = 0; row < buf_h; row++) {
                yTemp = buf_corner_y + row;
                yUse = yTemp;
                if (yTemp < 0){
                    yUse += 1;
                }
                if (yTemp > h - 1){
                    yUse -= 1;
                } 
                buffer[row * buf_w + lID] = in_values[yUse*gSizeX + xUse];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if ((y < h) && (x < w)){
        out_values[gID] = median9(buffer[(buf_y-1)*buf_w + (buf_x-1)],
                                  buffer[(buf_y-1)*buf_w + (buf_x)],
                                  buffer[(buf_y-1)*buf_w + (buf_x+1)],
                                  buffer[(buf_y)*buf_w + (buf_x-1)],
                                  buffer[(buf_y)*buf_w + (buf_x)],
                                  buffer[(buf_y)*buf_w + (buf_x+1)],
                                  buffer[(buf_y+1)*buf_w + (buf_x-1)],
                                  buffer[(buf_y+1)*buf_w + (buf_x)],
                                  buffer[(buf_y+1)*buf_w + (buf_x+1)]);
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
