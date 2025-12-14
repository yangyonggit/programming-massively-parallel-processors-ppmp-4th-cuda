#pragma once

#ifdef __CUDACC__
__host__ __device__
#endif
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}
