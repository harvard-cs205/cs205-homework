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
        // YOUR CODE HERE
        c_real = coords_real[x][y];
        c_imag = coords_imag[x][y];
        z_real = 0;
        z_imag = 0;
        for( iter=0; iter<=max_iter; ++iter)
            if magnitude_squared(z) > 4:
                break
                z = z * z + c
            out_counts[i, j] = iter
    }
}
