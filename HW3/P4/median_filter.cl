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
    int tmpx;
    int tmpy;
    // Load into local buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.
    //Use buf_w number of threads to update values in local buffer
    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
            tmpx = buf_corner_x+idx_1D;
            tmpy = buf_corner_y+row;
            //The first if clause copies from global buffer to local
            //The rest handle the halo
            if(tmpy >=0 && tmpy < h && tmpx >= 0 && tmpx < w){
              buffer[row * buf_w + idx_1D] = in_values[tmpy * w + tmpx];
            }else if(tmpx == -1 && tmpy == -1){
              buffer[row * buf_w + idx_1D] = in_values[tmpx+1];
            }else if(tmpx == w && tmpy == -1){
              buffer[row * buf_w + idx_1D] = in_values[tmpx-1];
            }else if(tmpx == -1 && y == h){
              buffer[row * buf_w + idx_1D] = in_values[(tmpy - 1) * w];
            }else if(tmpx == w && tmpy == h){
              buffer[row * buf_w + idx_1D] = in_values[(tmpy-1) * w + tmpx-1];
            }else if(tmpx < 0){
              buffer[row * buf_w + idx_1D] = in_values[tmpy * w];
            }else if(tmpx >= w){
              buffer[row * buf_w + idx_1D] = in_values[tmpy * w + tmpx-1];
            }else if(tmpy < 0){
              buffer[row * buf_w + idx_1D] = in_values[tmpx];
            }else if(tmpy >= h){
              buffer[row * buf_w + idx_1D] = in_values[(tmpy-1) * w + tmpx];
            }
                
        }

    barrier(CLK_LOCAL_MEM_FENCE);
    //Only compute median for non-halo pixels
    if(x < w && y < h){
      out_values[y * w + x] = median9(buffer[(buf_y-1)*buf_w+buf_x-1],buffer[(buf_y-1)*buf_w+buf_x],buffer[(buf_y-1)*buf_w+buf_x+1], \
                          buffer[buf_y*buf_w+buf_x-1],buffer[buf_y*buf_w+buf_x],buffer[buf_y*buf_w+buf_x+1],\
                          buffer[(buf_y+1)*buf_w+buf_x-1],buffer[(buf_y+1)*buf_w+buf_x],buffer[(buf_y+1)*buf_w+buf_x+1]);
    }
    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    //median9 takes 9 floats and returns median


    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
}
