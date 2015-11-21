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
    float z_real, z_imag, z_real_new, z_imag_new;
    float mag2;
    int iter;

    if ((x < w) && (y < h)) {
        iter = 1;
        c_real = coords_real[y*w + x];
        c_imag = coords_imag[y*w + x];
        z_real = c_real;
        z_imag = c_imag;
        mag2 = z_real*z_real + z_imag*z_imag; 
        while ((mag2 < 4) && (iter < max_iter)){
            z_real_new = z_real*z_real - z_imag*z_imag + c_real;
            z_imag_new = 2*z_real*z_imag + c_imag;
            mag2 = z_real_new*z_real_new + z_imag_new*z_imag_new;
            z_real = z_real_new;
            z_imag = z_imag_new;
            iter = iter + 1;
        }
    out_counts[y*w + x] = iter;
    }
}
