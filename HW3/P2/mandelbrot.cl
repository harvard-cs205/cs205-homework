__kernel void
mandelbrot(__global __read_only float *coords_real,
           __global __read_only float *coords_imag,
           __global __write_only int *out_counts,
           int w, int h, int max_iter)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float z_real, z_imag, c_real, c_imag, z_real1;
    int iter = 0;

    if ((x < w) && (y < h)) {
        z_real = 0;
        z_imag = 0;
        c_real = coords_real[y*w + x];
        c_imag = coords_imag[y*w + x];
        while ( ((z_real*z_real + z_imag*z_imag) <= 4) && (iter <= max_iter) ) {        
          z_real1 = (z_real*z_real - z_imag*z_imag) + c_real;
          z_imag = (2*z_real*z_imag) + c_imag;
          z_real = z_real1;
          iter++;
        }
     }

    out_counts[y*w + x] = iter;
}
