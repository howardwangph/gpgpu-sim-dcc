#pragma once

#include "cuda-sim/memory.h"
#include "cuda-sim/ptx_sim.h"

//Class definition for aggregated block group
class agg_block_group_t {
public:

    agg_block_group_t(dim3 agg_dim, 
        dim3 block_dim, kernel_info_t * kernel, addr_t p_mem_base, int kernelq_entry_id, ptx_thread_info *thread ) {
        m_agg_block_group_dim = agg_dim;
        m_agg_block_dim = block_dim;
        m_kernel = kernel;
        m_param_mem = new memory_space_impl<256>("param", 256);
	m_param_mem_base = p_mem_base; //bddream - DKPL
	kernel_queue_entry_id = kernelq_entry_id;
	p_thd = thread;
    }
    dim3 get_agg_dim() const {
        return m_agg_block_group_dim;
    }

    dim3 get_block_dim() const {
        return m_agg_block_dim;
    }

    size_t get_num_blocks() const {
        return m_agg_block_group_dim.x *
            m_agg_block_group_dim.y *
            m_agg_block_group_dim.z;
    }

    class memory_space * get_param_memory() const {
        return m_param_mem;
    }

    // bddream - DKPL
    addr_t get_param_memory_base() const {
       return m_param_mem_base;
    }

    int get_kernel_queue_entry_id() const {
	return kernel_queue_entry_id;
    }

    ptx_thread_info * get_parent_thd() const {
	return p_thd;
    }

private:
    dim3 m_agg_block_group_dim; // aggregated block group dim, similar to kernel grid dim
    dim3 m_agg_block_dim; // aggregated block dim
    kernel_info_t * m_kernel; // the native kernel to be aggregated to
    class memory_space * m_param_mem; // parameter buffer

    addr_t m_param_mem_base; // bddream - DKPL
    int kernel_queue_entry_id;
    ptx_thread_info *p_thd; // pointer to parent thread; for parent-child dependency

};
