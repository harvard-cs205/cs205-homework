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
    float z_real, z_imag, z_r_tmp;
    float mag;
    int iter;

    if ((x < w) && (y < h)) 
    {
        // YOUR CODE HERE
        c_real = coords_real[y * w + x];
        c_imag = coords_imag[y * w + x];
        
        // Initialize z = c
        z_real = c_real;
        z_imag = c_imag;

        // Initialize iter and mag
        iter = 0;
        mag = (z_real * z_real) + (z_imag * z_imag);
        
        // With no sqrt, compare against 4
        while (mag < 4 && (iter < max_iter)) {
          z_r_tmp = (z_real * z_real) - (z_imag * z_imag) + c_real;
          z_imag = (2 * z_real * z_imag) + c_imag;
          z_real = z_r_tmp;
          
          // Update mag and iter
          mag = (z_real*z_real) + (z_imag * z_imag);
          iter ++;
        }
    
    // Write out to memeory
    out_counts[y * w + x] = iter;
      
    }
}

