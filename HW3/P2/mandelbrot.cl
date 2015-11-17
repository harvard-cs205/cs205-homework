__kernel void
mandelbrot(__global __read_only float *coords_real,
           __global __read_only float *coords_imag,
           __global __write_only int *out_counts,
           int w, int h, int max_iter)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float c_real, c_imag;
    float z_real, z_imag;

    z_real = 0.0;
    z_imag = 0.0;
    c_real = coords_real[x + y*w];
    c_imag = coords_imag[x + y*w];
    int iter;

    // Make sure we are within bounds
    if ((x < w) && (y < h)) {
        // YOUR CODE HERE
        for(iter = 0; iter < max_iter; ++iter){
          if (z_real*z_real + z_imag*z_imag > 4)
            break;

          // Create temporary z
          float z_real_tmp = z_real*z_real - z_imag*z_imag;
          float z_imag_tmp = 2*z_real*z_imag;
          z_real = z_real_tmp + c_real;
          z_imag = z_imag_tmp + c_imag;
        }

    }
    out_counts[x+ y*w] = iter;
}
