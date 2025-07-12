#ifndef __GEMMINI_RUNTIME_H__
#define __GEMMINI_RUNTIME_H__

#include "gemmini_driver.h"

#define CONV2D_MS       256
#define CONV2D_GEMM_MS  85 //64
#define CONV3D_MS       40 //32
#define CONV3D_GEMM_MS  13 //8
#define MAXPOOL_MS      256
#define MAXPOOL_GEMM_MS MAXPOOL_MS
#define RELU_MS         256
#define MM_MS           128
#define MM_GEMM_MS      MM_MS

#define K_SIZE          3
#define P_SIZE          2

class GemminiRT {
public:

    GemminiRT() {}
    ~GemminiRT() {}

    bool conv2d(
        unsigned int m_size,
        unsigned int k_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool conv2dGemm(
        unsigned int m_size,
        unsigned int k_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool conv3d(
        unsigned int m_size,
        unsigned int k_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool conv3dGemm(
        unsigned int m_size,
        unsigned int k_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool maxPool(
        unsigned int m_size,
        unsigned int k_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool maxPoolGemm(
        unsigned int m_size,
        unsigned int k_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool relu(
        unsigned int m_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool mm(
        unsigned int m_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);

    bool mmGemm(
        unsigned int m_size,
        unsigned long &sw_time,
        unsigned long &hw_time,
        unsigned long &mem_time);
private:
    GemminiDriver gemminiDrv;
};

#endif
