#include "median9.h"

// 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo){   

    //Initial variable definitions for the problem
    //this code is obtained from the load halo example done in class

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    // OBTAIN BUFFER BY LOADING RELEVANT LABELS TO LOVAL BUFFER WITH HALO
    if (idx_1D<buf_w){

        int row,mXX,mYY,hNew,wNew,mat_index,buf_index;
        // GO OVER EACH ROW AND GO DOWN IN EACH COLUMN TO OBTAIN THE VALUES OF TH
        // BUFFER FOR EACH INDIVIDUAL NODE
        for(row=0;row<buf_h;row++){
            //GET THE XX AND YY, ROW AND COLUMN RESPECTIVE CORNERS WITH ADJUSTMENTS
            mYY=buf_corner_y+row;
            mXX=buf_corner_x+idx_1D;
            //OBTAIN THE NEW INDECES FOR THE HEIGHT AND THE WIDTH
            hNew=h-1;
            wNew=w-1;

            //GET THE INDECES FOR THE BUFFER THAN TO BE LOADED INTO THE BUFFER
            mat_index=min(max(0, mYY), hNew) * w + min(max(0, mXX), wNew);
            buf_index=row * buf_w + idx_1D;
            //LOAD DESIRED VALUES FROM GLOBAL MEMORY INTO THE BUFFER FOR FURTHER COMPUTATION
            buffer[buf_index] = in_values[mat_index];
        }

    }

    // if (idx_1D < buf_w)
    //     for (row = 0; row < buf_h; row++) {
    //         buffer[row * buf_w + idx_1D] = 
    //             get_clamped_value(labels,
    //                               w, h,
    //                               buf_corner_x + idx_1D, buf_corner_y + row);
    //     }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...
    //
    // Should only use buffer, buf_x, buf_y

    // write output
    // if ((y < img_h) && (x < img_w)) // stay in bounds
    //     output[y * img_w + x] = \
    //         buffer[buf_y * buf_w + buf_x];
    //STATEMENT FOR SMOOTHENING THE PIRURE WITHIN THE SPACE THAT WE CARE ABOUT
	if ((y < h) && (x < w)) {
        //THIS WRITES THE VALUE OUT AS DESIRED AND DESCRIBED IN THE .H FILE
	    out_values[y * w + x] = median9(buffer[(buf_y - 1) * buf_w + (buf_x - 1)], buffer[(buf_y - 1) * buf_w + buf_x], buffer[(buf_y - 1) * buf_w + (buf_x + 1)], 
	                                    buffer[buf_y * buf_w       + (buf_x - 1)], buffer[buf_y * buf_w       + buf_x], buffer[buf_y * buf_w       + (buf_x + 1)], 
	                                    buffer[(buf_y + 1) * buf_w + (buf_x - 1)], buffer[(buf_y + 1) * buf_w + buf_x], buffer[(buf_y + 1) * buf_w + (buf_x + 1)]);
	}
}
	


    // Note: It may be easier for you to implement median filtering
    // without using the local buffer, first, then adjust your code to
    // use such a buffer after you have that working.


    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.


    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.


    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.

