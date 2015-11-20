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
    float z_new_r = 0;
    float z_new_i = 0;
    z_real = 0;
    z_imag = 0;

    c_real = coords_real[w*y + x];
    c_imag = coords_imag[w*y + x];

    if ((x < w) && (y < h)) {
        // YOUR CODE HERE
        iter = 0;
    	for (int i = 0; i < max_iter; ++i) {
    		if (z_real*z_real + z_imag * z_imag > 4.0) {
    			break;
    		}
    		iter++;
    		// z = z * z + c
    		z_new_r = z_real * z_real - z_imag * z_imag;
    		z_new_i = 2.0 * z_real * z_imag;
    		z_new_i += c_imag;
    		z_new_r += c_real;
    		z_real = z_new_r;
    		z_imag = z_new_i;
    	}
    	out_counts[w * y + x] = iter;
  	}
}
