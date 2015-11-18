//######################
//#
//# Submission by Kendrick Lo (Harvard ID: 70984997) for
//# CS 205 - Computing Foundations for Computational Science (Prof. R. Jones)
//# 
//# Homework 3 - Problem 2
//#
//######################

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
        
        int offset = y * w + x;

        c_real = coords_real[offset];
        c_imag = coords_imag[offset];
        z_real = 0;
        z_imag = 0;
        iter = 0;

        for(int i=0; i<max_iter; i++) {

            if ((z_real * z_real + z_imag * z_imag) > 4)
                break;        
            float temp = z_real;

            // zr * zr + cr - zi * zi
            z_real = (temp * temp) + c_real - (z_imag * z_imag);
            // 2 * zr * zi + ci
            z_imag = 2 * temp * z_imag + c_imag;
            iter += 1;
        }
        out_counts[offset] = iter;
    }
}

// observations:
// buffers are one-dimensional -> need offset
// need to know C: for loops, if statements; semi-colons, print statements

// The platforms detected are:
// ---------------------------
// Apple Apple version: OpenCL 1.2 (May 10 2015 19:38:45)
// The devices detected on platform Apple are:
// ---------------------------
// Intel(R) Core(TM) i7-5557U CPU @ 3.10GHz [Type: CPU ]
// Maximum clock Frequency: 3100 MHz
// Maximum allocable memory size: 4294 MB
// Maximum work group size 1024
// ---------------------------
// Intel(R) Iris(TM) Graphics 6100 [Type: GPU ]
// Maximum clock Frequency: 1100 MHz
// Maximum allocable memory size: 402 MB
// Maximum work group size 256
// ---------------------------
// This context is associated with  2 devices
// The queue is using the device: Intel(R) Iris(TM) Graphics 6100
// 1074.93164 Million Complex FMAs in 0.05021424 seconds, 21406.9084786 million Complex FMAs / second
