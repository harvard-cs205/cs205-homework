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
        z_real = 0;
        z_imag = 0;
        c_real = coords_real[x + (y * w)];
        c_imag = coords_imag[x + (y * w)];

        for (iter=0; iter < max_iter; iter++){
          if (z_real * z_real + z_imag * z_imag > 4)
            break;

          float z_real_update = z_real * z_real - z_imag * z_imag;
          float z_imag_update = 2 * z_imag * z_real;
          z_real = z_real_update + c_real;
          z_imag = z_imag_update + c_imag;
        }

        out_counts[(y * w) + x] = iter;
        
    }
}
