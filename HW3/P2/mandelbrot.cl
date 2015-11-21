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
    float newZr, newZi;

    int iter;

    // if inside of the boundaries then do the computation
    if ((x < w) && (y < h)) {
 		
 		// initializing the variables
        z_real = 0;
        z_imag = 0;
        c_real = coords_real[(y * w) + x];
        c_imag = coords_imag[(y * w) + x];
        
        // generate the mandelbrot set
        for (iter = 0; iter < max_iter; iter++) {

            //computing the magnitude, if greater than 4, quit the computation
            if (z_real*z_real + z_imag*z_imag > 4.0) {
                break;
            }
            
            // get the new values for each array or real and imaginary separately
            newZr = (z_real*z_real) - (z_imag*z_imag);
            newZi = 2*(z_imag*z_real);
            z_real = newZr + c_real;
            z_imag = newZi + c_imag;
        
        }
        // Transfer iteration data to output
        out_counts[(y * w) + x] = iter;
    }
}
