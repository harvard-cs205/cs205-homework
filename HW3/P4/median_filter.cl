#include "median9.h"

// FETCH function used below
float FETCH(__global __read_only float *image, int img_w, int img_h, int x, int y)
{
    // Pixel value from original image to return
    float val;
    
    // If out of bounds, correct to the edge of the image
    if (x < 0) x = 0;
    if (x >= img_w) x = img_w - 1;
    if (y < 0) y = 0;
    if (y >= img_h) y = img_h - 1;
    
    // Take pixel value from the image
    val = image[img_w * y + x];    
    
    return val;
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
    // The first several lines of code are taken almost directly from load_halo.cl in 
    //     the harvard-cs205 OpenCL-examples folder:
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    
    // Global position of pixel
    const int glob_x = get_global_id(0);
    const int glob_y = get_global_id(1);
    
    // Local position of pixel within workgroup
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    // Global coordinates of upper left corner of buffer in the image space
    const int buf_corner_x = glob_x - local_x - halo;
    const int buf_corner_y = glob_y - local_y - halo;
    
    // Local coordinates of pixel in the local buffer
    const int buf_x = local_x + halo;
    const int buf_y = local_y + halo;
    
    // 1D index of thread within work group
    const int local_index = local_y * get_local_size(0) + local_x;
    
    // Row within work group
    int row;
    
    // Load into buffer (with 1-pixel halo).
    // Globally out-of-bounds pixels are replaced with the nearest valid pixel's value.
    
    if (local_index < buf_w) {
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + local_index] = 
            // Use the FETCH() function defined above to pull out the appropriate pixel from the image.
            FETCH(in_values, w, h, buf_corner_x + local_index,
                buf_corner_y + row);
        }
    }

    // Wait for all the threads to complete this step
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    
    // Each thread in the valid region (x < w, y < h) writes back its 3x3 neighborhood median using the median9 function.
    if ((glob_y < h) && (glob_x < w)) {
        out_values[glob_y * w + glob_x] = 
        median9(buffer[(buf_y-1)*buf_w + buf_x-1], buffer[(buf_y-1)*buf_w + buf_x], buffer[(buf_y-1)*buf_w + buf_x+1],
                buffer[(buf_y)*  buf_w + buf_x-1], buffer[(buf_y)  *buf_w + buf_x], buffer[(buf_y)  *buf_w + buf_x+1],
                buffer[(buf_y+1)*buf_w + buf_x-1], buffer[(buf_y+1)*buf_w + buf_x], buffer[(buf_y+1)*buf_w + buf_x+1]);    
    }
}
