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
    float new_z_real;
    int iter;

    if ((x < w) && (y < h)) {
        // YOUR CODE HERE
        // implementing mandelbrot here 

        // initialize
        iter = 0;
        z_real =0;
        z_imag = 0;
        c_real = coords_real[w*x + y];
        c_imag = coords_imag[w*x + y];
        while((z_real*z_real+ z_imag*z_imag <=4) \
        &&(iter <= max_iter)){
          // Similar to AVX implemtation
          new_z_real = (z_real*z_real - z_imag*z_imag) \
                      + c_real;
          z_imag = (2 * z_real* z_imag) + c_imag;
          z_real = new_z_real;
          iter = iter + 1;
        }



        ;
    out_counts[x*w + y] = iter;
    }
}
