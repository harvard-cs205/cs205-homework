//#include "median9.h"


#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) < (b)) ? (b) : (a))

#define cas(a, b) tmp = min(a, b); b = max(a, b); a = tmp

inline float median9(float s0, float s1, float s2,
                     float s3, float s4, float s5,
                     float s6, float s7, float s8)
{
    // http://a-hackers-craic.blogspot.com/2011/05/3x3-median-filter-or-branchless.html
    float tmp;
        
    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);

    cas(s0, s1);
    cas(s3, s4);
    cas(s6, s7);

    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);

    cas(s3, s6);
    cas(s4, s7);
    cas(s5, s8);
    cas(s0, s3);

    cas(s1, s4);
    cas(s2, s5);
    cas(s3, s6);

    cas(s4, s7);
    cas(s1, s3);

    cas(s2, s6);

    cas(s2, s3);
    cas(s4, s6);

    cas(s3, s4);

    return s4;
}

//Return image value at point (x,y), where the indicies are relative to the global image. If the indicies are out of bounds, choose closest in bounds value instead. 
inline float fetch_point(__global __read_only float *img, 
                   int w, int h,
                   int x, int y)
{
    float out_img;
    while (x<0) {
        x++;
    }
    while (x>w-1){
        x--;
    }
    while (y<0) {
        y++;
    }
    while (y>h-1){
        y--;
    }
    out_img=img[w*y+x];

    return out_img;

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

    //store calculated median here before transferring 
    float out_buffer; 
    
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

    const int idx_1D = ly * get_local_size(0) + lx;

    int row;

    if (x < w && y < h){
    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.
        if (idx_1D < buf_w) {
            for (row = 0; row < buf_h; row++) {
                buffer[row * buf_w + idx_1D] = \
                    fetch_point(in_values, w, h,
                          buf_corner_x + idx_1D,
                          buf_corner_y + row);
          }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 3x3 median for each pixel in core (non-halo) pixels
        //
        // We've given you median9.h, and included it above, so you can
        // use the median9() function.

        //core point is buffer[(ly+halo)*buf_w+(lx+halo)]. For example, for (lx,ly)=(0,0), their value in the buffer should be (1,1). In the variables below, bc stands for "buffer core."
        int bc_x=lx+halo;
        int bc_y=ly+halo;
        out_buffer = median9(buffer[(bc_y-1)*buf_w+bc_x-1],
                             buffer[(bc_y-1)*buf_w+bc_x],
                             buffer[(bc_y-1)*buf_w+bc_x+1],
                             buffer[(bc_y)*buf_w+bc_x-1],
                             buffer[(bc_y)*buf_w+bc_x],
                             buffer[(bc_y)*buf_w+bc_x+1],
                             buffer[(bc_y+1)*buf_w+bc_x-1],
                             buffer[(bc_y+1)*buf_w+bc_x],
                             buffer[(bc_y+1)*buf_w+bc_x+1]);


        out_values[y*w+x]=out_buffer;
  }
    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
}
