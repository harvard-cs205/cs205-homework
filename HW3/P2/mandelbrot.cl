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
    float z_real, z_imag, z_real_t, z_imag_t;
    int iter, index;

    if ((x < w) && (y < h)) {
        // YOUR CODE HERE
		index =  y * w + x;
		c_real = coords_real[index];
		c_imag = coords_imag[index];  
		z_real = 0;
		z_imag = 0;
        for(int iter = 0; iter < max_iter; iter++) {
            if ( z_real * z_real + z_imag * z_imag  > 4.0f ) {
                break;
			}
			// (a+bi)*(c+di)=(ac-bd)+(bc+ad)i
			// (a+bj)+(c+dj)=(a+c)+(b+d)i
			z_real_t = z_real * z_real - z_imag * z_imag;
			z_imag_t = z_imag * z_real + z_real * z_imag;
			z_real = z_real_t + c_real;
			z_imag = z_imag_t + c_imag;
			out_counts[index] = iter;
		}
    }
}
