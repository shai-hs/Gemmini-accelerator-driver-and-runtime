#ifndef __GEMMINI_DRIVER_HPP__
#define __GEMMINI_DRIVER_HPP__

typedef enum GemminiDevAOP_
{
    op_conv2d,
    op_conv2d_gemm,
    op_conv3d,
    op_conv3d_gemm,
    op_maxpool,
    op_maxpool_gemm,
    op_relu,
    op_mm,
    op_mm_gemm
} GemminiDevAOP;

class GemminiDriver {
public:

    GemminiDriver();
    ~GemminiDriver() {}

    void *staticAlloc(size_t bytes);
    void staticFree(void *ptr);
    void callGemminiDevA(float *m, float *k, float *o,
                         unsigned int m_size, unsigned int k_size,
                         GemminiDevAOP opcode);
private:
    char *static_ptr;
};

#endif
