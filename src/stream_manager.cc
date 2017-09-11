// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "stream_manager.h"
#include "gpgpusim_entrypoint.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"

//Po-Han DCC
#include "cuda-sim/cuda_device_runtime.h"

unsigned CUstream_st::sm_next_stream_uid = 0;

CUstream_st::CUstream_st() 
{
    m_pending = false;
    m_uid = sm_next_stream_uid++;
    pthread_mutexattr_init(&m_attr);
    pthread_mutexattr_settype(&m_attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&m_lock, &m_attr);
//    pthread_mutex_init(&m_lock,NULL);
}


bool CUstream_st::empty()
{
    pthread_mutex_lock(&m_lock);
    bool empty = m_operations.empty();
    pthread_mutex_unlock(&m_lock);
    return empty;
}

bool CUstream_st::busy()
{
    pthread_mutex_lock(&m_lock);
    bool pending = m_pending;
    pthread_mutex_unlock(&m_lock);
    return pending;
}

void CUstream_st::synchronize() 
{
    // called by host thread
    bool done=false;
    do{
        pthread_mutex_lock(&m_lock);
        done = m_operations.empty();
        pthread_mutex_unlock(&m_lock);
    } while ( !done );
}

void CUstream_st::push( const stream_operation &op )
{
    // called by host thread
    pthread_mutex_lock(&m_lock);
    m_operations.push_back( op );
    pthread_mutex_unlock(&m_lock);
}

void CUstream_st::record_next_done()
{
    // called by gpu thread
    pthread_mutex_lock(&m_lock);
    assert(m_pending);
    m_operations.pop_front();
    m_pending=false;
    pthread_mutex_unlock(&m_lock);
}


stream_operation CUstream_st::next()
{
    // called by gpu thread
    pthread_mutex_lock(&m_lock);
    m_pending = true;
    stream_operation result = m_operations.front();
    pthread_mutex_unlock(&m_lock);
    return result;
}

void CUstream_st::cancel_front()
{
    pthread_mutex_lock(&m_lock);
    assert(m_pending);
    m_pending = false;
    pthread_mutex_unlock(&m_lock);

}

void CUstream_st::print(FILE *fp)
{
    pthread_mutex_lock(&m_lock);
    fprintf(fp,"GPGPU-Sim API:    stream %u has %zu operations\n", m_uid, m_operations.size() );
    std::list<stream_operation>::iterator i;
    unsigned n=0;
    for( i=m_operations.begin(); i!=m_operations.end(); i++ ) {
        stream_operation &op = *i;
        fprintf(fp,"GPGPU-Sim API:       %u : ", n++);
        op.print(fp);
        fprintf(fp,"\n");
    }
    pthread_mutex_unlock(&m_lock);
}


bool stream_operation::do_operation( gpgpu_sim *gpu )
{
    if( is_noop() ) 
        return true;

    assert(!m_done && m_stream);
    if(g_debug_execution >= 3)
       printf("GPGPU-Sim API: stream %u performing ", m_stream->get_uid() );
    switch( m_type ) {
    case stream_memcpy_host_to_device:
        if(g_debug_execution >= 3)
            printf("memcpy host-to-device\n");
        gpu->memcpy_to_gpu(m_device_address_dst,m_host_address_src,m_cnt);
        m_stream->record_next_done();
        break;
    case stream_memcpy_device_to_host:
        if(g_debug_execution >= 3)
            printf("memcpy device-to-host\n");
        gpu->memcpy_from_gpu(m_host_address_dst,m_device_address_src,m_cnt);
        m_stream->record_next_done();
        break;
    case stream_memcpy_device_to_device:
        if(g_debug_execution >= 3)
            printf("memcpy device-to-device\n");
        gpu->memcpy_gpu_to_gpu(m_device_address_dst,m_device_address_src,m_cnt); 
        m_stream->record_next_done();
        break;
    case stream_memcpy_to_symbol:
        if(g_debug_execution >= 3)
            printf("memcpy to symbol\n");
        gpgpu_ptx_sim_memcpy_symbol(m_symbol,m_host_address_src,m_cnt,m_offset,1,gpu);
        m_stream->record_next_done();
        break;
    case stream_memcpy_from_symbol:
        if(g_debug_execution >= 3)
            printf("memcpy from symbol\n");
        gpgpu_ptx_sim_memcpy_symbol(m_symbol,m_host_address_dst,m_cnt,m_offset,0,gpu);
        m_stream->record_next_done();
        break;
    case stream_kernel_launch:
        if( m_sim_mode ) { //Functional Sim
            printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n", m_kernel->get_uid(), m_kernel->name().c_str() );
                m_kernel->print_parent_info();
        	gpu->set_cache_config(m_kernel->name());
            gpu->functional_launch( m_kernel );
        }
        else { //Performance Sim
            if( gpu->can_start_kernel() && m_kernel->m_launch_latency == 0) {
            	printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n", m_kernel->get_uid(), m_kernel->name().c_str() );
                    m_kernel->print_parent_info();
//                printf("running kernels size %d, max_concurrent_kernels %d\n", gpu->get_running_kernels_size(), gpu->get_config().get_max_concurrent_kernel());
//                gpu->print_running_kernels_stats();
            	gpu->set_cache_config(m_kernel->name());
                gpu->launch( m_kernel );
            }
            else {
                if(m_kernel->m_launch_latency)
                    m_kernel->m_launch_latency--;
                if(g_debug_execution >= 3)
//                if(g_debug_execution >= 1)
        	        printf("kernel %d: \'%s\', latency %u not ready to transfer to GPU hardware scheduler\n", 
                        m_kernel->get_uid(), m_kernel->name().c_str(), m_kernel->m_launch_latency);
                return false;    
            }
        }
        break;
    case stream_event: {
        printf("event update\n");
        time_t wallclock = time((time_t *)NULL);
        m_event->update( gpu_tot_sim_cycle, wallclock );
        m_stream->record_next_done();
        } 
        break;
    default:
        abort();
    }
    m_done=true;
    fflush(stdout);
    return true;
}

void stream_operation::print( FILE *fp ) const
{
    fprintf(fp," stream operation " );
    switch( m_type ) {
    case stream_event: fprintf(fp,"event"); break;
    case stream_kernel_launch: fprintf(fp,"kernel"); break;
    case stream_memcpy_device_to_device: fprintf(fp,"memcpy device-to-device"); break;
    case stream_memcpy_device_to_host: fprintf(fp,"memcpy device-to-host"); break;
    case stream_memcpy_host_to_device: fprintf(fp,"memcpy host-to-device"); break;
    case stream_memcpy_to_symbol: fprintf(fp,"memcpy to symbol"); break;
    case stream_memcpy_from_symbol: fprintf(fp,"memcpy from symbol"); break;
    case stream_no_op: fprintf(fp,"no-op"); break;
    }
}

stream_manager::stream_manager( gpgpu_sim *gpu, bool cuda_launch_blocking ) 
{
    m_gpu = gpu;
    m_service_stream_zero = false;
    m_cuda_launch_blocking = cuda_launch_blocking;
    pthread_mutex_init(&stm_m_lock,NULL);
}

bool stream_manager::operation( bool * sim)
{
    bool check=check_finished_kernel();
    pthread_mutex_lock(&stm_m_lock);
//    if(check)m_gpu->print_stats();
    stream_operation op =front();
    if(!op.do_operation( m_gpu )) //not ready to execute
    {
        //cancel operation
        if( op.is_kernel() ) {
            unsigned grid_uid = op.get_kernel()->get_uid();
            m_grid_id_to_stream.erase(grid_uid);
        }
        op.get_stream()->cancel_front();

    }
    pthread_mutex_unlock(&stm_m_lock);
    //pthread_mutex_lock(&m_lock);
    // simulate a clock cycle on the GPU
    return check;
}

bool stream_manager::check_finished_kernel()
{
    unsigned grid_uid = m_gpu->finished_kernel();
    bool check=register_finished_kernel(grid_uid);
    return check;
}

unsigned stream_manager::gpu_can_start_kernel(){
	return m_gpu->can_start_kernel();
}

bool stream_manager::register_finished_kernel(unsigned grid_uid)
{
    // called by gpu simulation thread
    if(grid_uid > 0){
        CUstream_st *stream = m_grid_id_to_stream[grid_uid];
        kernel_info_t *kernel = stream->front().get_kernel();
        assert( grid_uid == kernel->get_uid() );

        //Jin: should check children kernels for CDP
        if(kernel->is_finished()) {
            std::ofstream kernel_stat("kernel_stat.txt", std::ofstream::out | std::ofstream::app);
            kernel_stat<< " kernel " << grid_uid << ": " << kernel->name();
//printf("XD4\n");
            if(kernel->get_parent()){
                kernel_stat << ", parent " << kernel->get_parent()->get_uid() <<
                ", launch " << kernel->launch_cycle;
            kernel_stat<< ", start " << kernel->start_cycle <<
                ", end " << kernel->end_cycle << ", retire " << gpu_sim_cycle + gpu_tot_sim_cycle << "\n";
	    } else {
		    extern unsigned long long max_concurrent_device_kernel, concurrent_device_kernel;
		    if(max_concurrent_device_kernel < concurrent_device_kernel) max_concurrent_device_kernel = concurrent_device_kernel;
		    concurrent_device_kernel = 0;
	    }
//printf("XD5\n");

	    //turn off the flag after parent kernel finished
	    extern bool child_running;
	    if(is_target_parent_kernel(kernel)){
		printf("Cycle %llu: child kernels finished.\n", gpu_sim_cycle+gpu_tot_sim_cycle);
		child_running = false;
	    }

            printf("kernel %d finishes, retires from stream %d\n", grid_uid, stream->get_uid());
            kernel_stat.flush();
            kernel_stat.close();
            stream->record_next_done();
            m_grid_id_to_stream.erase(grid_uid);
            kernel->notify_parent_finished();
            delete kernel;
            return true;
        }
    }

    return false;
}

stream_operation stream_manager::front() 
{
    // called by gpu simulation thread
    stream_operation result;
//    if( concurrent_streams_empty() )
    m_service_stream_zero = true;
    if( m_service_stream_zero ) {
        if( !m_stream_zero.empty() && !m_stream_zero.busy() ) {
                result = m_stream_zero.next();
                if( result.is_kernel() ) {
                    unsigned grid_id = result.get_kernel()->get_uid();
                    m_grid_id_to_stream[grid_id] = &m_stream_zero;
                }
        } else {
            m_service_stream_zero = false;
        }
    }
    
    if(!m_service_stream_zero)
    {
        std::list<struct CUstream_st*>::iterator s, s_end;
	s_end = m_streams.end();
//        if(m_streams.size() != 0){
        for( s=m_streams.begin(); s != s_end/*m_streams.end()*/; s++) {
            CUstream_st *stream = *s;
            if( !stream->empty()/*busy()*/ && !stream->busy()/*empty()*/ ) {
                result = stream->next();
                if( result.is_kernel() ) {
                    unsigned grid_id = result.get_kernel()->get_uid();
                    m_grid_id_to_stream[grid_id] = stream;
                }
                break;
            }
        }
//        }
    }
    return result;
}

void stream_manager::add_stream( struct CUstream_st *stream )
{
    // called by host thread
    pthread_mutex_lock(&stm_m_lock);
    m_streams.push_back(stream);
    pthread_mutex_unlock(&stm_m_lock);
}

void stream_manager::destroy_stream( CUstream_st *stream )
{
    // called by host thread
    pthread_mutex_lock(&stm_m_lock);
    while( !stream->empty() )
        ; 
    std::list<CUstream_st *>::iterator s;
//        if(m_streams.size() != 0){
    for( s=m_streams.begin(); s != m_streams.end(); s++ ) {
        if( *s == stream ) {
            m_streams.erase(s);
            break;
        }
    }
//        }
    delete stream; 
    pthread_mutex_unlock(&stm_m_lock);
}

bool stream_manager::has_stream( CUstream_st *stream ){
	if(std::find(m_streams.begin(), m_streams.end(), stream) != m_streams.end())
		return true;
	else
		return false;
}

bool stream_manager::concurrent_streams_empty()
{
    bool result = true;
    // called by gpu simulation thread
    std::list<struct CUstream_st *>::iterator s, s_end;
    if (!m_streams.empty()){
	unsigned idx = 0;
	s_end = m_streams.end();
	for( s=m_streams.begin(); s!=s_end/*m_streams.end()*/;s++, idx++ ) {
	    struct CUstream_st *stream = *s;
	    if( !stream->empty() ) {
		//            stream->print(stdout);
		result = false;
	    }
	}
    }
    return result;
}

bool stream_manager::empty_protected()
{
    bool result = true;
    pthread_mutex_lock(&stm_m_lock);
    if( !concurrent_streams_empty() )
        result = false;
    if( !m_stream_zero.empty() )
        result = false;
/*    extern std::list<dcc_kernel_distributor_t> g_cuda_dcc_kernel_distributor;
    extern bool g_dyn_child_thread_consolidation;
    if( g_dyn_child_thread_consolidation && !g_cuda_dcc_kernel_distributor.empty() ){
	result = false;
	printf("DCC: there are %d child kernels in the kernel distrobutor, prevent GPGPUsim from halting\n", g_cuda_dcc_kernel_distributor.size());
    }*/
    pthread_mutex_unlock(&stm_m_lock);
    return result;
}

bool stream_manager::empty()
{
//    bool result = true;
    if( !concurrent_streams_empty() ) 
	    return false;
//        result = false;
    if( !m_stream_zero.empty() ) 
	    return false;
//        result = false;
//    return result;
    return true;
}


void stream_manager::print( FILE *fp)
{
    pthread_mutex_lock(&stm_m_lock);
    print_impl(fp);
    pthread_mutex_unlock(&stm_m_lock);
}
void stream_manager::print_impl( FILE *fp)
{
    fprintf(fp,"GPGPU-Sim API: Stream Manager State\n");
    std::list<struct CUstream_st *>::iterator s;
//        if(m_streams.size() != 0){
    for( s=m_streams.begin(); s!=m_streams.end();++s ) {
        struct CUstream_st *stream = *s;
        if( !stream->empty() ) 
            stream->print(fp);
    }
//        }
    if( !m_stream_zero.empty() ) 
        m_stream_zero.print(fp);
}

void stream_manager::push( stream_operation op )
{
    struct CUstream_st *stream = op.get_stream();

    // block if stream 0 (or concurrency disabled) and pending concurrent operations exist
    bool block= !stream || m_cuda_launch_blocking;
    while(block) {
        pthread_mutex_lock(&stm_m_lock);
        block = !concurrent_streams_empty();
        pthread_mutex_unlock(&stm_m_lock);
    };

    pthread_mutex_lock(&stm_m_lock);
    if( stream && !m_cuda_launch_blocking ) {
        stream->push(op);
    } else {
        op.set_stream(&m_stream_zero);
        m_stream_zero.push(op);
    }
    if(g_debug_execution >= 3)
       print_impl(stdout);
    pthread_mutex_unlock(&stm_m_lock);
    if( m_cuda_launch_blocking || stream == NULL ) {
        unsigned int wait_amount = 100; 
        unsigned int wait_cap = 100000; // 100ms 
        while( !empty_protected() ) {
            // sleep to prevent CPU hog by empty spin
            // sleep time increased exponentially ensure fast response when needed 
            usleep(wait_amount); 
            wait_amount *= 2; 
            if (wait_amount > wait_cap) 
               wait_amount = wait_cap; 
        }
    }
}

//Jin: aggregated blocks support
kernel_info_t * stream_manager::find_grid(function_info * entry, kernel_info_t * parent_grid = NULL, unsigned parent_block_idx = 0)
{
    kernel_info_t * grid = NULL;
    if(parent_grid == NULL){
	grid = m_stream_zero.find_grid(entry, NULL, 0);
    }else{
	grid = m_stream_zero.find_grid(entry, parent_grid, parent_block_idx);
    }
    if(grid != NULL)
	return grid;

    //        if(m_streams.size() != 0){
    for(auto s=m_streams.begin(); s!=m_streams.end();++s ) {
	if(parent_grid == NULL){
	    grid = (*s)->find_grid(entry, NULL, 0);
	}else{
	    grid = (*s)->find_grid(entry, parent_grid, parent_block_idx);
	}
        if(grid != NULL)
            return grid;
    }
//        }

    return NULL;
}

kernel_info_t * CUstream_st::find_grid(function_info * entry, kernel_info_t *parent_grid = NULL, unsigned parent_block_idx = 0) 
{
    kernel_info_t * grid = NULL;

    pthread_mutex_lock(&m_lock);

    for( auto op=m_operations.begin(); op!=m_operations.end(); op++ ) {
        if(op->is_kernel()) {
            kernel_info_t * kernel = op->get_kernel();
	    if(parent_grid == NULL){
		if(entry == kernel->entry() && !kernel->is_finished() ) {
		    grid = kernel;
		    break;
		}
	    } else {
		if( !kernel->is_finished() && kernel->get_parent() == parent_grid && kernel->get_parent_block_idx() == parent_block_idx ){
		    grid = kernel;
		    break;
		}
	    }
        }
    }

    pthread_mutex_unlock(&m_lock);

    return grid;
}

unsigned stream_manager::stream_count(){
   return m_streams.size();
}
