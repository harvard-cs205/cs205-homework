

//Load in the median function header file
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

    //////////////////////Problem Comentary///////////////////////

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



    //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // with in workgroup, so less than buffer
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 8
    

    //////////////////////loop to build Buffer///////////////////////

    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
        //Iterate down each colum, using a row iterator
        for (int row = 0; row < buf_h; row++) 
       {

          int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
          int max_y = buf_corner_y + row; //stepping by rows adjust y
          int new_h = h - 1; // height index
          int new_w = w - 1; // width index

          // Load the values into the buffer
          // This is a read from global memory global read
          // Each thread is loading values into the buffer down columns
          buffer[row * buf_w + idx_1D] = in_values[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        }

    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);



//////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {
        // Create new index, 'idx,' for reference within function call
        // to median9
        int idx = (lx + 1) + (2 * halo + get_local_size(0)) * (ly + 1); 

        // calculate and save new median for the appropirate neighbor values
        float ans = median9(buffer[idx - buf_w - 1], buffer[idx - buf_w], buffer[idx - buf_w + 1], buffer[idx - 1], buffer[idx], buffer[idx + 1], buffer[idx + buf_w - 1], buffer[idx + buf_w], buffer[idx + buf_w + 1]);
        
        // Write out the new median value, saved under 'ans'
        // This is a global write
        out_values[y * w + x] = ans;

      } 

}
