#include "median9.h"

//get the value
//just like FETCH macro
float get_values(__global __read_only float *in_values,
           int img_w, int img_h,
           int mem_x, int mem_y)
{
    //check the boundary
    //to see if it's in the halo including area
    //if mem_x is out of boundary, either to make it to 0, to the last boundary value
    //we want to find the nearest valid pixel's values
    if(mem_x < 0){
        mem_x = 0;
    }
    if(mem_x >= img_w){
        mem_x = img_w - 1;
    }
    if(mem_y >= img_h){
        mem_y = img_h - 1;
    }
    if(mem_y < 0){
        mem_y = 0;
    }

    return in_values[mem_y * img_w + mem_x];

}


// 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int img_w, int img_h,
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


    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.


    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

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

    //read row by row, as we have talked in the class
    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
                get_values(in_values, img_w, img_h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Should only use buffer, buf_x, buf_y.

    // write output
    if ((y < img_h) && (x < img_w)){ // stay in bounds
    // here we read from the buffer and write to output
    // here the median9 function to calculate 9 values.
    // the offset has the format of  y * width + x_index
        out_values[y * img_w + x] = median9(buffer[(buf_y-1)*buf_w + buf_x - 1],
                                        buffer[(buf_y-1)*buf_w + buf_x],
                                        buffer[(buf_y-1)*buf_w + buf_x + 1],
                                        buffer[(buf_y)*buf_w + buf_x-1],
                                        buffer[(buf_y)*buf_w + buf_x],
                                        buffer[(buf_y)*buf_w + buf_x+1],
                                        buffer[(buf_y+1)*buf_w + buf_x-1],
                                        buffer[(buf_y+1)*buf_w + buf_x],
                                        buffer[(buf_y+1)*buf_w + buf_x+1]
                                        );



    }


}
