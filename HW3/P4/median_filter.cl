#include "median9.h"

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

inline int to_buf_1d(int lx, int ly, const int halo, int buf_w)
{
    return (ly + halo) * buf_w + (lx + halo);
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

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int local_size = get_local_size(0);

    // exclude pixels outside of bounds
    if (x >= w || y >= h)
    {
        return;
    }

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // each thread loads 3 x 3 neighbors into corresponding buffer locations
    // loop over the size of the filter kernel which is (2 * halo + 1)
    for (int i = -halo; i <= halo; i++)
    for (int j = -halo; j <= halo; j++)
    {
        int x_clamped, y_clamped, lx_clamped, ly_clamped;

        if (x + i < 0 || x + i >= w)
        {
            x_clamped = x;
            lx_clamped = lx;
        }
        else
        {
            x_clamped = x + i;
            lx_clamped = lx + i;
        }

        if (y + j < 0 || y + j >= h)
        {
            y_clamped = y;
            ly_clamped = ly;
        }
        else
        {
            y_clamped = y + j;
            ly_clamped = ly + j;
        }

        // get 1d index of image to copy from
        int image_idx = y_clamped * w + x_clamped;

        // get 1d index of buffer to write into
        int buf_idx = to_buf_1d(lx_clamped, ly_clamped, halo, buf_w);

        buffer[buf_idx] = in_values[image_idx];
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    int buf_idx[9];
    int b = 0;
    for (int i = -halo; i <= halo; i++)
    for (int j = -halo; j <= halo; j++)
    {
        int lx_clamped = (x + i < 0 || x + i >= w) ? lx : lx + i;
        int ly_clamped = (y + j < 0 || y + j >= h) ? ly : ly + j;

        buf_idx[b] = to_buf_1d(lx_clamped, ly_clamped, halo, buf_w);
        b++;
    }

    float filtered = median9(buffer[buf_idx[0]], buffer[buf_idx[1]], buffer[buf_idx[2]],
                             buffer[buf_idx[3]], buffer[buf_idx[4]], buffer[buf_idx[5]],
                             buffer[buf_idx[6]], buffer[buf_idx[7]], buffer[buf_idx[8]]);

    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    // get 1d index of output image to write to
    out_values[y * w + x] = filtered;
}
