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
        // Initialize calculation
        z_real = 0;
        z_imag = 0;
        c_real = coords_real[(y * w) + x];
        c_imag = coords_imag[(y * w) + x];
        
        // Calculate mandelbrot set
        for (iter = 0; iter < max_iter; iter++) {
            
            if (z_real*z_real + z_imag*z_imag > 4.0) {
                break;
            }
            
            float zr_new = (z_real*z_real) - (z_imag*z_imag);
            float zi_new = 2*(z_imag*z_real);
            z_real = zr_new + c_real;
            z_imag = zi_new + c_imag;
        
        }
        // Transfer iteration data to output
        out_counts[(y * w) + x] = iter;
        
    }
}