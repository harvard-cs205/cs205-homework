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
    float temp;
    int iter;

    if ((x < w) && (y < h)) {
        //initialization of the variables
        iter = 0;
        z_real = 0;
        z_imag = 0;
        //get value from global: 1D indexing is w*y + x
        c_real = coords_real[w*y + x];
        c_imag = coords_imag[w*y + x];

        //loop for mandelbrot
        while(((z_real*z_real+z_imag*z_imag)<=4)&&(iter<=max_iter)) {
            //update real part in a temporary value
            temp = z_real*z_real -z_imag*z_imag + c_real;
            //update imaginary part
            z_imag = 2*z_real*z_imag + c_imag;
            z_real = temp;
            iter++;
        }
    }
    //put value back in global the 1D indexing is w*y + x
    out_counts[w*y + x] = iter;
}
