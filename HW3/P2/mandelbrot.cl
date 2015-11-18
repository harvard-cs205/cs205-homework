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

    if ((x < w) && (y < h)) {
        for(int iter=0; iter < max_iter; iter++){

        }
    }
    out_counts[x, y] = 0;
}
