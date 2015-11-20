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
    
    // Initialize the Local Position
    // x = rows, y = cols
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    
    // Now the Global Position
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    // Local buffer top left pixel reference
    const int buf_corner_x = gx - lx - halo;
    const int buf_corner_y = gy - ly - halo;

    // Local buffer location of current pixel
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.
    
    // Stay within image boundary
    if (idx_1D < buf_w){

        // Loop through rows
        for (int row=0; row < buf_h; row++){
            
            // Save global reference based on buffer location
            int x_ref = buf_corner_x + idx_1D;
            int y_ref = buf_corner_y + row;

            // Load buffer based on global location
            // If y/x < 0, take 0 and if > h/w take h/w
            // Kevin Chen helped me visual the min, max function 
            buffer[row * buf_w + idx_1D] = in_values[ min( max(0, y_ref), (h-halo) ) * w \
                                                    + min( max(0, x_ref), (w-halo) ) ];
        }

    }

    // Ensure all buffer loads are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.

    float filter_result;
    // Ensure that global id stays within w, h
    if ( (gx<w) && (gy<h) ){

        // Pass 9 buffer value window to Median Filter
        filter_result = median9(
                    buffer[(buf_y-1) * buf_w + buf_x], buffer[(buf_y-1) * buf_w + (buf_x+1)], 
                    buffer[buf_y * buf_w + (buf_x+1)], buffer[(buf_y+1) * buf_w + (buf_x+1)], 
                    buffer[(buf_y+1) * buf_w + buf_x], buffer[(buf_y+1) * buf_w + (buf_x-1)], 
                    buffer[(buf_y) * buf_w + (buf_x-1)], buffer[(buf_y-1) * buf_w + (buf_x-1)],
                    buffer[buf_y * buf_w + buf_x]);

        // Each thread in the valid region (x < w, y < h) should write
        // back its 3x3 neighborhood median.
        out_values[gy * w + gx] = filter_result;
    }

    
}
