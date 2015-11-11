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
    float tmp_real;
    int iter, idx;

    if ((x < w) && (y < h)) {
        // YOUR CODE HERE
        idx = y*w + x;
        z_real = 0.0;
        z_imag = 0.0;
        c_real = coords_real[idx];
        c_imag = coords_imag[idx];
        iter = 0;
        while ((iter < max_iter) && (z_real*z_real + z_imag*z_imag <= 4.0)) {
            // z = z * z + c
            // z_real = z_real**2 - z_imag**2 + c_real
            // z_imag = 2 * z_real * z_imag * i + c_imag * i
            tmp_real = z_real * z_real - z_imag * z_imag + c_real;
            z_imag = 2.0 * z_real * z_imag + c_imag;
            z_real = tmp_real;
            iter++;
        }
        out_counts[idx] = iter;
    }
}
