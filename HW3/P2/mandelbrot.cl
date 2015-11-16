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
    float z_real, z_imag, z_real_tmp;
    int iter;

    if ((x < w) && (y < h)) {
        z_real = 0;
        z_imag = 0;
        c_real = coords_real[y * w + x];
        c_imag = coords_imag[y * w + x];

        for (iter = 0; iter < max_iter; iter++) {
            if ((z_real * z_real + z_imag * z_imag) > 4)
                break;

            z_real_tmp = (z_real * z_real) - (z_imag * z_imag) + c_real;
            z_imag = 2 * z_real * z_imag + c_imag;
            z_real = z_real_tmp;
        }

        out_counts[y * w + x] = iter;
    }
}
