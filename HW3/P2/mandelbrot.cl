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
    int iter;

    if ((x < w) && (y < h)) {
      x = y * w + x;

      z_real = x;
      z_imag = y;
      c_real = x;
      c_imag = y;
      iter = 0;
        while (abs(z) < 2) && (iter < 511) {
            z_real = z_real*z_real - z_imag*z_imag + c_real;
            z_imag = 2*z_real*z_imag + c_imag
            iter++;
          }

    }
}
