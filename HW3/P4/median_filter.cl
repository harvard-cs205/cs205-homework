#include "median9.h"

static float
get_clamped_value(__global __read_only float *in_values,
                  int w, int h,
                  int x, int y, int halo)
/*# This function is modified from P5. 
  # First, check that buffer coordinates are within global bounds.
  # If not, assign out-of-bounds coordinates the value of nearest valid pixel.
*/
{
    if (x<0) {
      if (y<0) {
        return in_values[0];
      } else if (y>=h) {
        return in_values[(y-halo) * w + (x + halo)];
      } else {
        return in_values[y * w + (x + halo)];
      }
    } else if (y<0) {
      if (x>=w) {
        return in_values[(y+halo) * w + (x-halo)];
      } else { //#we already account for x<0&&y<0 in the (x<0) block above
        return in_values[(y+halo) * w + x];
      }
    } else if (x>=w) {
      if (y>=h) {
        return in_values[(y-halo) * w + (x-halo)];
      } else { //#y<0 && x>=w is covered above
        return in_values[y * w + (x-halo)];
      }
    } else if (y>=h) { # all out-of-bounds x cases are covered, we only need the in-bounds x case here
      return in_values[(y-halo) * w + x];
    } else {
      return in_values[y * w + x];
    }
}

//# 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo)
{

    //# Drawing heavily from P5 and load_halo.cl

    //# Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    //# Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    //# coordinates of the upper left corner of the buffer in image
    //# space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    //# coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    //# 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;
    
    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = 
                get_clamped_value(in_values,
                                  w, h,
                                  buf_corner_x + idx_1D, buf_corner_y + row, halo);
        }
    }

    //# Make sure all threads reach the next part after the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    //# Return median values for pixels within global boundaries
    if ((x < w) && (y < h)) {
      out_values[y * w + x] = median9(buffer[(buf_y-1) * buf_w + buf_x -1], buffer[(buf_y-1) * buf_w + buf_x], buffer[(buf_y-1) * buf_w + buf_x +1],
                                      buffer[buf_y * buf_w + buf_x -1],     buffer[buf_y * buf_w + buf_x],     buffer[buf_y * buf_w + buf_x +1],
                                      buffer[(buf_y+1) * buf_w + buf_x -1], buffer[(buf_y+1) * buf_w + buf_x], buffer[(buf_y+1) * buf_w + buf_x +1]
                                    );
    }

}
