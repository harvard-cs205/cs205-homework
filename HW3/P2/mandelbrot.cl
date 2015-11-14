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
    float z_real, z_imag, old_z_real, old_z_imag;
    float magnitude;
    int iter;

    // initialize z to 0
    z_real = z_imag = 0;
    c_real = coords_real[y * w + x];
    c_imag = coords_imag[y * w + x];

    if ((x < w) && (y < h)) {
      for (iter = 0; iter < max_iter; iter++)
      {
        magnitude = z_real * z_real + z_imag * z_imag;

        // if magnitude is > 4, break
        if (magnitude > 4)
        {
          break;
        }

        // save old z values for computation
        old_z_real = z_real;
        old_z_imag = z_imag;

        // update z
        // z = a + bi
        // c = c + di
        // z = (a + bi)(a + bi) + (c + di)
        // z = a^2 + 2(a * bi) + (bi)^2 + c + di
        // z_real = a^2 - b^2 + c
        // z_imag = 2(a * bi) + di

        z_real = old_z_real * old_z_real - old_z_imag * old_z_imag + c_real;
        z_imag = 2 * (old_z_real * old_z_imag) + c_imag;
      }
    }

    out_counts[y * w + x] = iter;

}
