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
    int lid_x = get_local_id(0);
    int lid_y = get_local_id(1);
    int lsize_x = get_local_size(0);
    int lsize_y = get_local_size(1);
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int gsize_x = get_global_size(0);
    int gsize_y = get_global_size(1);
    
    // Note: It may be easier for you to implement median filtering
    // without using the local buffer, first, then adjust your code to
    // use such a buffer after you have that working.
    
//    // This is the non-local buffer version
//    out_values[ind2d1d(gid_x, gid_y, gsize_x)] =
//    median9(in_values[ind2d1d(get_app_pix(gid_x - 1, w), get_app_pix(gid_y - 1, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x + 0, w), get_app_pix(gid_y - 1, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x + 1, w), get_app_pix(gid_y - 1, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x - 1, w), get_app_pix(gid_y + 0, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x + 0, w), get_app_pix(gid_y + 0, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x + 1, w), get_app_pix(gid_y + 0, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x - 1, w), get_app_pix(gid_y + 1, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x + 0, w), get_app_pix(gid_y + 1, h), w)],
//            in_values[ind2d1d(get_app_pix(gid_x + 1, w), get_app_pix(gid_y + 1, h), w)]);
    
    
    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = gid_x - lid_x - halo;
    const int buf_corner_y = gid_y - lid_y - halo;
    
    // coordinates of our pixel in the local buffer
    const int buf_x = lid_x + halo;
    const int buf_y = lid_y + halo;
    
    // 1D index of thread within our work-group
    const int idx_1D = lid_y * lsize_x + lid_x;
    
    
    // Load one row at a time by splitting work among threads.
    // Each row has buf_w=2*halo+lsize_x elements (here, buf_w=10)
    if (idx_1D < buf_w) {
        for(int row_i = 0; row_i < buf_h; row_i++) {
            buffer[ind2d1d(idx_1D, row_i, buf_w)] =
            in_values[ind2d1d(get_app_pix(buf_corner_x + idx_1D, w),
                              get_app_pix(buf_corner_y + row_i, h), w)];
        }
    }
    // Synchronize threads so that we don't start processing buffer
    // before buffer is loaded.
    barrier(CLK_LOCAL_MEM_FENCE);

    
    out_values[ind2d1d(gid_x, gid_y, gsize_x)] =
    median9(buffer[ind2d1d(buf_x - 1, buf_y - 1, buf_w)],
            buffer[ind2d1d(buf_x + 0, buf_y - 1, buf_w)],
            buffer[ind2d1d(buf_x + 1, buf_y - 1, buf_w)],
            buffer[ind2d1d(buf_x - 1, buf_y + 0, buf_w)],
            buffer[ind2d1d(buf_x + 0, buf_y + 0, buf_w)],
            buffer[ind2d1d(buf_x + 1, buf_y + 0, buf_w)],
            buffer[ind2d1d(buf_x - 1, buf_y + 1, buf_w)],
            buffer[ind2d1d(buf_x + 0, buf_y + 1, buf_w)],
            buffer[ind2d1d(buf_x + 1, buf_y + 1, buf_w)]);


    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
}

// Function to convert pair of 2d indices to a single 1d index
int ind2d1d(int ind_x, int ind_y, int w) {
    return ind_y * w + ind_x;
}

// Function that returns a padded pixel index if on the edge
int get_app_pix(int x, int w) {
    if (x < 0) {
        return 0;
    }else if(x >= w) {
        return w - 1;
    }else {
        return x;
    }
}

