#include <cstddef>
#include <cstdint>
#include "gemmini_driver.h"

#define GEMMINI_DEV_A_ADDR_CTRL 0x40000000
#define STATIC_MEM_BASE_ADDR 0x40001000
#define STATIC_MEM_TOTL_SIZE 0x3ffff000
#define ALIGN_SIZE 0x40

GemminiDriver::GemminiDriver()
{
    static_ptr = (char *) STATIC_MEM_BASE_ADDR;
}

void * GemminiDriver::staticAlloc(size_t bytes)
{
    if (static_ptr + bytes < (char *) STATIC_MEM_BASE_ADDR + STATIC_MEM_TOTL_SIZE)
    {
        char *aux_ptr = static_ptr;
        
        size_t offset = bytes / ALIGN_SIZE * ALIGN_SIZE;
        if (bytes > offset)
            offset += ALIGN_SIZE;
        
        static_ptr += offset;
        
        return aux_ptr;
    }

    return NULL;
}

void GemminiDriver::staticFree(void *ptr)
{
    return;
}

void GemminiDriver::callGemminiDevA(
    float *m,
    float *k,
    float *o,
    unsigned int m_size,
    unsigned int k_size,
    GemminiDevAOP opcode)
{
    uint64_t *gemmini_dev_a_ctrl = (uint64_t *) GEMMINI_DEV_A_ADDR_CTRL;

    gemmini_dev_a_ctrl[0] = (uint64_t) m;
    gemmini_dev_a_ctrl[1] = (uint64_t) k;
    gemmini_dev_a_ctrl[2] = (uint64_t) o;
    gemmini_dev_a_ctrl[3] = (uint64_t) m_size;
    gemmini_dev_a_ctrl[4] = (uint64_t) k_size;
    gemmini_dev_a_ctrl[5] = (uint64_t) opcode;
    
    gemmini_dev_a_ctrl[6] = (uint64_t) 0x1;
    
    while(!gemmini_dev_a_ctrl[7]);
}
