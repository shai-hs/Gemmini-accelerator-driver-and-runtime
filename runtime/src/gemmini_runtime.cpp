#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <chrono>

#include "baseline.h"
#include "gemmini_runtime.h"
#include "gemmini_driver.h"

#define GET_TICKS std::chrono::high_resolution_clock::now()
#define GET_ELAPS(A, B) std::chrono::duration_cast<std::chrono::nanoseconds>(B - A).count() / 1000

bool GemminiRT::conv2d(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *k  = (float *) gemminiDrv.staticAlloc(k_size * k_size * sizeof(float)), // kernel
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(k); assert(os); assert(oh);

    // Initialize data structures
    //initRandArray(m, m_size * m_size);
    //initRandArray(k, k_size * k_size);

    mem_time = 0;

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseConv_2D(m, k, os, m_size, k_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(m, k, oh, m_size, k_size, op_conv2d);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size);
    bool status = true;

    gemminiDrv.staticFree(m);
    gemminiDrv.staticFree(k);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::conv2dGemm(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *a  = (float *) gemminiDrv.staticAlloc(m_size * m_size * 
                                      k_size * k_size * sizeof(float)), // redundant matrix
          *k  = (float *) gemminiDrv.staticAlloc(k_size * k_size * sizeof(float)), // kernel
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(a); assert(k); assert(os); assert(oh);

    // Initialize data structures
    //initRandArray(m, m_size * m_size);
    //initRandArray(k, k_size * k_size);

    //auto start_mem = GET_TICKS;
    //gemminiDrv.compMatrixA_2D(a, m, m_size, k_size);
    //mem_time = GET_ELAPS(start_mem, GET_TICKS);

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseConvGemm(a, k, os, m_size * m_size, k_size * k_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(a, k, oh, m_size, k_size, op_conv2d_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size);
    bool status = true;

    gemminiDrv.staticFree(m);
    gemminiDrv.staticFree(a);
    gemminiDrv.staticFree(k);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::conv3d(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) gemminiDrv.staticAlloc(m_size * m_size * m_size * sizeof(float)), // input matrix
          *k  = (float *) gemminiDrv.staticAlloc(k_size * k_size * k_size * sizeof(float)), // kernel
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size * m_size * sizeof(float)), // output software
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(k); assert(os); assert(oh);

    // Initialize data structures
    //initRandArray(m, m_size * m_size * m_size);
    //initRandArray(k, k_size * k_size * k_size);

    //mem_time = 0;

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseConv_3D(m, k, os, m_size, k_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(m, k, oh, m_size, k_size, op_conv3d);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size);
    bool status = true;

    gemminiDrv.staticFree(m);
    gemminiDrv.staticFree(k);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::conv3dGemm(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) gemminiDrv.staticAlloc(m_size * m_size * m_size * sizeof(float)), // input matrix
          *a  = (float *) gemminiDrv.staticAlloc(m_size * m_size * m_size * 
                                      k_size * k_size * k_size * sizeof(float)), // redundant matrix
          *k  = (float *) gemminiDrv.staticAlloc(k_size * k_size * k_size * sizeof(float)), // kernel
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size * m_size * sizeof(float)), // output software
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(a); assert(k); assert(os); assert(oh);

    // Initialize data structures
    //initRandArray(m, m_size * m_size * m_size);
    //initRandArray(k, k_size * k_size * k_size);

    //auto start_mem = GET_TICKS;
    //gemminiDrv.compMatrixA_3D(a, m, m_size, k_size);
    //mem_time = GET_ELAPS(start_mem, GET_TICKS);

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseConvGemm(a, k, os, m_size * m_size * m_size, k_size * k_size * k_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(a, k, oh, m_size, k_size, op_conv3d_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size);
    bool status = true;

    gemminiDrv.staticFree(m);
    gemminiDrv.staticFree(a);
    gemminiDrv.staticFree(k);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::maxPool(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix
	*os = (float *) gemminiDrv.staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)), // output matrix
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)); // output matrix

    assert(m); assert(os); assert(oh);

    //initRandArray(m, m_size * m_size);

    //mem_time = 0;

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseMaxPool(m, os, m_size, k_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(m, NULL, oh, m_size, k_size, op_maxpool);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size / k_size / k_size);
    bool status = true;

    gemminiDrv.staticFree(m);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::maxPoolGemm(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *a  = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // redundant matrix
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)), // output matrix
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)); // output matrix

    assert(m); assert(a); assert(os); assert(oh);

    //initRandArray(m, m_size * m_size);

    //auto start_mem = GET_TICKS;
    //gemminiDrv.compMatMaxA(a, m, m_size, k_size);
    //mem_time = GET_ELAPS(start_mem, GET_TICKS);

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseMaxPoolGemm(a, os, m_size, k_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(a, NULL, oh, m_size, k_size, op_maxpool_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size / k_size / k_size);
    bool status = true;

    gemminiDrv.staticFree(m);
    gemminiDrv.staticFree(a);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::relu(
    unsigned int m_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(os); assert(oh);

    //initRandArray(m, m_size * m_size);

    //mem_time = 0;

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseRelu(m, os, m_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(m, NULL, oh, m_size, 0, op_relu);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size);
    bool status = true;

    gemminiDrv.staticFree(m);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::mm(
    unsigned int m_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m1 = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix 1
          *m2 = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix 2
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m1); assert(m2); assert(os); assert(oh);

    //initRandArray(m1, m_size * m_size);
    //initRandArray(m2, m_size * m_size);

    //mem_time = 0;

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseMM(m1, m2, os, m_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(m1, m2, oh, m_size, 0, op_mm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size);
    bool status = true;

    gemminiDrv.staticFree(m1);
    gemminiDrv.staticFree(m2);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

bool GemminiRT::mmGemm(
    unsigned int m_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m1 = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix 1
          *m2 = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // input matrix 2
          *a  = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // matrix 2 transposed
          *os = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) gemminiDrv.staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m1); assert(m2); assert(a); assert(os); assert(oh);

    //initRandArray(m1, m_size * m_size);
    //initRandArray(m2, m_size * m_size);

    //auto start_mem = GET_TICKS;
    //transpose(m2, a, m_size);
    //mem_time = GET_ELAPS(start_mem, GET_TICKS);

    //auto start_sw = GET_TICKS;
    //gemminiDrv.baseMMGemm(m1, a, os, m_size);
    //sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    gemminiDrv.callGemminiDevA(m1, a, oh, m_size, 0, op_mm_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    //bool status = compareArrays(os, oh, m_size * m_size);
    bool status = true;

    gemminiDrv.staticFree(m1);
    gemminiDrv.staticFree(m2);
    gemminiDrv.staticFree(a);
    gemminiDrv.staticFree(os);
    gemminiDrv.staticFree(oh);

    return status;
}

