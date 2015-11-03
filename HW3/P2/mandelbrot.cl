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
    float z_real_sq, z_imag_sq, mag_sq;
    int offset, iter;

    if ((x < w) && (y < h)) {
        offset = (y * w) + x;
        c_real = coords_real[offset];
        c_imag = coords_imag[offset];
        z_real = 0;
        z_imag = 0;
        for (iter=0; iter < max_iter; iter++) {
            z_real_sq = z_real * z_real;
            z_imag_sq = z_imag * z_imag;
            mag_sq = z_real_sq + z_imag_sq;
            if (mag_sq > 4.0) {
                break;
            }
            z_imag = (2 * z_real * z_imag) + c_imag;
            z_real = z_real_sq - z_imag_sq + c_real;
        }
        out_counts[offset] = iter;
    }
}
