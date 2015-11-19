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

    float new_z_real;
    iter = 0;

    if ((x < w) && (y < h)) {

      c_real = coords_real[x + y * w];
      c_imag = coords_imag[x + y * w];
      z_real = 0.0;
      z_imag = 0.0;
        
        // perform loop
        for(int i = 0; i < max_iter; ++i) {

          // compute z^2 + c
          new_z_real = z_real * z_real - z_imag * z_imag + c_real;
          z_imag = 2. * z_real * z_imag + c_imag;
          z_real = new_z_real;

          // check if amplitude is less than 4, if so inc!
          if(z_imag * z_imag + z_real * z_real < 4.) {
            iter += 1;
          }
        }

        out_counts[x + y * w] = iter;
    }


}
