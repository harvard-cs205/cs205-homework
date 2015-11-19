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
        z_real = 0;
        z_imag = 0;
        c_real = x;
        c_imag = y;
        iter = 0;
        while (z_real*z_real + z_imag*z_imag < 4 and iter < max_iter) {
            z_real = z_real*z_real + z_imag*z_imag + c_real;
            z_imag = z_real*z_imag + c_imag;
            iter += 1;
        }
        out_counts[y*w + x];
    }
}
