// This exercise was discussed with Andrew Petschek


#include "median9.h"

// Creating a function that fills the buffer with the right values (like in PB 5)
float
get_buff(__global __read_only float *in_values,
                  int w, int h,
                  int x, int y)
// Checking multiple cases:
{   if (x<0){
      if (y<0) {return in_values[0];} // top left corner of the halo
      else if (y >= h) {return in_values[(h-1) * w];} //bottom left corner of the halo
      else {return in_values[y*w];} //left side of the halo
    }

    else if (x>=w){
      if (y<0) {return in_values[w-1];} //top right corner of the halo
      else if (y>=h) {return in_values[(h-1) * w + w - 1];} //bottom right corner of the halo
      else {return in_values[(w-1) + y*w];} // right side of the halo
    }

    else{
      if (y<0) {return in_values[x];} //top side of the halo
      else if (y>=h) {return in_values[x + (h-1) * w];} //lower side of the halo
      else {return in_values[x + w*y];} // Regular case (one without any exception)
    }
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

    // halo is the additional number of cells in one direction

    const int x = get_global_id(0);
    const int y = get_global_id(1);

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
  
    if (idx_1D < buf_w){
      for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = get_buff(in_values, w, h,buf_corner_x + idx_1D,buf_corner_y + row);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((y < h) && (x < w)) // Checking we are in bound
      //using the 8 values surronding (buf_x,buff_y)
        out_values[y * w + x] = median9(buffer[(buf_y-1) * buf_w + (buf_x-1)],
                                    buffer[(buf_y-1)  * buf_w + buf_x],
                                    buffer[(buf_y-1)  * buf_w + (buf_x+1)],
                                    buffer[buf_y * buf_w + (buf_x-1)],
                                    buffer[buf_y * buf_w + buf_x],
                                    buffer[buf_y * buf_w + (buf_x+1)],
                                    buffer[(buf_y+1) * buf_w + (buf_x-1)],
                                    buffer[(buf_y+1) * buf_w + buf_x],
                                    buffer[(buf_y+1) * buf_w + (buf_x+1)]);
  }     

