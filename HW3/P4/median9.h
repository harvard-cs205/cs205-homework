// median9.h - branchless median 

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) < (b)) ? (b) : (a))

#define cas(a, b) tmp = min(a, b); b = max(a, b); a = tmp

inline float median9(float s0, float s1, float s2,
                     float s3, float s4, float s5,
                     float s6, float s7, float s8)
{
    // http://a-hackers-craic.blogspot.com/2011/05/3x3-median-filter-or-branchless.html
    float tmp;
        
    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);

    cas(s0, s1);
    cas(s3, s4);
    cas(s6, s7);

    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);

    cas(s3, s6);
    cas(s4, s7);
    cas(s5, s8);
    cas(s0, s3);

    cas(s1, s4);
    cas(s2, s5);
    cas(s3, s6);

    cas(s4, s7);
    cas(s1, s3);

    cas(s2, s6);

    cas(s2, s3);
    cas(s4, s6);

    cas(s3, s4);

    return s4;
}
