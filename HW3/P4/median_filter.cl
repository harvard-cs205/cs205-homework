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
    // coordinates of output image (globally)
    int x = get_global_id(0);
    int y = get_global_id(1);

    // coordinates w.r.t. workgroup (locally)
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // coordinates of corners of local buffers
    int buf_corner_x = x-lx-halo;
    int buf_corner_y = y-ly-halo;
    
    // coordinates of pixel in local buffer
    int buf_x = lx+halo;
    int buf_y = ly+halo;

    // Convert 2D coordinates into 1D w.r.t. local buffer
    int idx_1D = ly*get_local_size(0)+lx;
    
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
    if(idx_1D<buf_w){
      // replace x_idx to stay in-bounds
      int x_idx = buf_corner_x+idx_1D;
      if(x_idx>=w) x_idx = w-1;
      else if(x_idx<0) x_idx = 0;
      // Save data of workgroup into local buffer by rows
      for(int r=0; r<buf_h; r++){
        // replace y_idx to stay in-bounds
        int y_idx = buf_corner_y+r;
        if(y_idx>=h) y_idx = h-1;
        else if(y_idx<0) y_idx = 0;
        buffer[r*buf_w+idx_1D] = in_values[y_idx*w+x_idx];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    float result = 
	median9(buffer[(buf_y-1)*buf_w+(buf_x-1)],buffer[(buf_y-1)*buf_w+(buf_x)],buffer[(buf_y-1)*buf_w+(buf_x+1)],
                buffer[(buf_y)*buf_w+(buf_x-1)],  buffer[(buf_y)*buf_w+(buf_x)], buffer[(buf_y)*buf_w+(buf_x+1)],
                buffer[(buf_y+1)*buf_w+(buf_x-1)],buffer[(buf_y+1)*buf_w+(buf_x)],buffer[(buf_y+1)*buf_w+(buf_x+1)]);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if(x<w && y<h) out_values[y*w+x] = result; 
}
