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

    float z_r_squared, z_i_squared, z_temp;
    int offset;

    if ((x < w) && (y < h)) {

        // Initialize
        offset = x * w + y;
        c_real = coords_real[offset];
        c_imag = coords_imag[offset];
        z_real = 0;
        z_imag = 0;

        // Update z based on magnitude
        for (iter = 0; iter < max_iter; iter++)
        {
            z_r_squared = z_real * z_real;
            z_i_squared = z_imag * z_imag;
            if (z_r_squared + z_i_squared > 4)
            {
                break;
            }
            
            z_imag = 2 * z_real * z_imag + c_imag;
            z_real = z_r_squared - z_i_squared + c_real;
        }
        out_counts[offset] = iter;
    }
}
