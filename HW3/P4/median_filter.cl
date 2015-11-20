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
  
    //The below code is referenced from load_halo.cl
    //which is mentioned in the problem as something we should
    // look at. https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl

      // Global position of output pixel (getting x and y because these are 2D)
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    // Same idea as above but for the local workgroup.
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    // Now lets get the leftmost and topmost x and y coordinates for 
    // the buffer
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // Figure out the exact location of where we are in the buffer
    // in terms of x and y coordinates (image-style indexed) 

    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    // Do the technique learned in class to determine where we are aka "ly rows down"
    // and lx columns to the right
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;
    int adjusted_x;
    int adjusted_y;

    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
          //Ok now we are getting into modified code from what was provided in load.cl

          //Here we check if the horizontal-index of the pixel we are looking at is outside the width 
          //(too far to right), and if it is we adjust our value for x to just be the boundary (width - 1)
          if (buf_corner_x + idx_1D >= w) {
            adjusted_x = w-1;
          }
          //Here we check if the pixel is too far to the left (less than 0) and if it is then
          //we adjust our x to be 0.
          else if (buf_corner_x + idx_1D < 0){
            adjusted_x = 0;
          }
          //Below, we are in bounds, so we can use the normal pixel as our index into buffer.
          else{
            adjusted_x = buf_corner_x + idx_1D;
          }
          //Similar to above, check if our current row is too high (below the image)
          if (buf_corner_y + row >= h){
            adjusted_y = h-1;
          }
          //Similar to above, check if our current row is too low (above the image)
          else if (buf_corner_y + row < 0){
            adjusted_y = 0;
          }
          else{
            adjusted_y = buf_corner_y + row;
          }
          //Set the element for this row from in_values
          buffer[row*buf_w + idx_1D] = in_values[adjusted_y*w + adjusted_x];
        }  

    barrier(CLK_LOCAL_MEM_FENCE);
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


    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

    if((x < w) && (y < h)){
      // use the median9 function from the previous pset. Basically here we want to pass
      // in the values in the buffer for the 3x3 square all around the pixel. These are the
      // pixels in the row above this pixel (first line of the function call), the row of the pixel(second line),
      // and the row below the pixel (last line).
      out_values[y*w + x] = median9(buffer[buf_y*buf_w - buf_w + buf_x - 1], buffer[buf_y*buf_w - buf_w + buf_x], buffer[buf_y*buf_w - buf_w + buf_x + 1], 
                                    buffer[buf_y*buf_w + buf_x - 1], buffer[buf_y*buf_w + buf_x], buffer[buf_y*buf_w + buf_x + 1], 
                                    buffer[buf_y*buf_w + buf_w + buf_x - 1], buffer[buf_y*buf_w + buf_w + buf_x], buffer[buf_y*buf_w + buf_w + buf_x + 1]);
    }           

}
