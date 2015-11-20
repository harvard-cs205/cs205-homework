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
        c_real = coords_real[x + y * w];
        c_imag = coords_imag[x + y * w];
        z_real = 0;
        z_imag = 0;

        for (iter = 0; iter < max_iter; ++iter) {
            if (z_real * z_real + z_imag * z_imag > 4.0) {
                break;
            }
            float temp = z_real * z_real - z_imag * z_imag + c_real;

            z_imag = (z_real * z_imag * 2) + c_imag;
            z_real = temp;
        }

        out_counts[x + y * w] = iter;
    }
}
