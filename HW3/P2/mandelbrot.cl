__kernel void
mandelbrot(__global __read_only float *coords_real,
           __global __read_only float *coords_imag,
           __global __write_only int *out_counts,
           int w, int h, int max_iter)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float c_real, c_imag, tmp_real;
    float z_real, z_imag, tmp_imag;
    int ind, iter;
    
    if ((x < w) && (y < h)) {
        // compute ind in coords 
        ind = y*w + x;
        c_real = coords_real[ind];
        c_imag = coords_imag[ind];
        // init z and iter
        z_real = 0.0; 
        z_imag = 0.0; 
        iter = 0;
        // check magnitude and iter
        while((iter < max_iter) && (z_real*z_real + z_imag*z_imag <= 4.0))
        {
            // update z
            tmp_real = z_real*z_real - z_imag*z_imag + c_real;
            tmp_imag = 2.0 * z_real * z_imag + c_imag;
            z_real = tmp_real;
            z_imag = tmp_imag;
            iter++;
        }
        out_counts[ind] = iter;
    }
}
