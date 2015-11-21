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
    float z_real, z_imag, z_real_temp;
    int iter;

    if ((x < w) && (y < h)) {
        // Get the coordinates that correspond to our current position
        // And also instantiate the z_real and z_imag values in case that is necessary
        z_real = 0;
        z_imag = 0;
        c_real = coords_real[x + y * w];
        c_imag = coords_imag[x + y * w];

        for (iter = 0; iter < max_iter; iter++){
            // If our value has already diverged, stop
            if ((z_real * z_real + z_imag * z_imag) > 4)
                break;

            // We want to calculate z' = z * z + c
            // This translates to, for z = x + iy
            // Re(z') = x^2 - y^2 + Re(c)
            // Im(z') = 2xy + Im(c)
            z_real_temp = z_real;
            z_real = c_real + (z_real * z_real) - (z_imag * z_imag);
            z_imag = 2 * z_real_temp * z_imag + c_imag;
        }

        out_counts[x + y * w] = iter;
    }
}

#OMG!!! SO FAST

