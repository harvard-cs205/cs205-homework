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
    float val;
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    const int idx_1D = ly * get_local_size(0) + lx;

    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            int tmp_x = idx_1D;
            int tmp_y = row;
            if(buf_corner_x + tmp_x < 0){
                tmp_x++;
            }
            else if(buf_corner_x + tmp_x > w-1){
                tmp_x--;
            }

            if(buf_corner_y + tmp_y < 0){
                tmp_y++;
            }
            else if(buf_corner_y + tmp_y > h-1){
                tmp_y--;
            }
            buffer[row * buf_w + idx_1D] = 
                    in_values[buf_corner_x + tmp_x + w*(buf_corner_y + tmp_y)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
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
    if((x < w) && (y < h)){
        float topL = buffer[buf_w*(buf_y-1)+buf_x-1];
        float topM = buffer[buf_w*(buf_y-1)+buf_x];
        float topR = buffer[buf_w*(buf_y-1)+buf_x+1];
        float midL = buffer[buf_y*buf_w+buf_x-1];
        float midM = buffer[buf_y*buf_w + buf_x];
        float midR = buffer[buf_y*buf_w +buf_x+1];
        float botL = buffer[buf_w*(buf_y+1)+buf_x-1];
        float botM = buffer[buf_w*(buf_y+1)+buf_x];
        float botR = buffer[buf_w*(buf_y+1)+buf_x+1];
        val = median9(topL, topM, topR, midL, midM, midR, botL, botM, botR);
        out_values[x+(w*y)] = val;
        }
    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
}
