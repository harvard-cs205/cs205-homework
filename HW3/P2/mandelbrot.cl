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
    float z_real, z_imag, zr_temp;
    int iter;

    z_real = z_imag = iter = 0.;
    c_real = coords_real[y * w + x];
    c_imag = coords_imag[y * w + x];

    if ((x < w) && (y < h)) {

        while (z_real * z_real + z_imag * z_imag < 4.0 && iter < max_iter) {
            zr_temp = z_real;
            z_real = z_real * z_real - z_imag * z_imag + c_real;
            z_imag = 2.0 * zr_temp * z_imag + c_imag;

            iter++;
        }

        out_counts[y * w + x] = iter;
    }

}
