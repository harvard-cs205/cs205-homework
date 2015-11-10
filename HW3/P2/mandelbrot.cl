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
    float z_old;
    int iter;

    if (x % 1000 == 0) {
        printf("%d \n", x);
    }

    if ((x < w) && (y < h)) {
        c_real = coords_real[y * w + x];
        c_imag = coords_imag[y * w + x];

        z_real = 0;
        z_imag = 0;

        for (iter = 0; iter < max_iter; iter++) {
            if ((z_real * z_real + z_imag * z_imag) > 4) {
                break;
            }

            z_old = z_real;
            z_real = z_real * z_real - z_imag * z_imag + c_real;
            z_imag = 2 * z_old * z_imag + c_imag;
        }

        out_counts[y * w + x] = iter;
    }
}
