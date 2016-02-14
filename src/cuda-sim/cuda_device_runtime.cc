//Jin: cuda_device_runtime.cc
//Defines CUDA device runtime APIs for CDP support

#include <iostream>
#include <map>

#define __CUDA_RUNTIME_API_H__

#include <builtin_types.h>
#include <driver_types.h>
#include "../gpgpu-sim/gpu-sim.h"
#include "cuda-sim.h"
#include "ptx_ir.h"
#include "../stream_manager.h"
#include "cuda_device_runtime.h"
#include "../agg_block_group.h"

//#define MAX_PARAM_BUFFER_SIZE 32768

#define DEV_RUNTIME_REPORT(a) \
  if( g_debug_execution ) { \
     std::cout << __FILE__ << ", " << __LINE__ << ": " << a << "\n"; \
     std::cout.flush(); \
  }

#if 0
//Po-Han: dynamic child-thread consolidation support
class dcc_kernel_distributor_t {

public:
   dcc_kernel_distributor_t() {}

   dcc_kernel_distributor_t(kernel_info_t * _kernel_grid, /*function_info * _kernel_entry,*/
     unsigned int _thread_count,
     void * _parameter_buffer):
      valid(false),
      candidate(false),
      launched(false),
      kernel_grid(_kernel_grid),
      thread_count(_thread_count),
      parameter_buffer(_parameter_buffer) {}

   bool valid, candidate, launched;
   kernel_info_t *kernel_grid;
   unsigned int thread_count;
   void *parameter_buffer;
};
#endif

class device_launch_config_t {

public:
   device_launch_config_t() {}

   device_launch_config_t(dim3 _grid_dim,
     dim3 _block_dim,
     unsigned int _shared_mem,
     function_info * _entry):
      grid_dim(_grid_dim),
      block_dim(_block_dim),
      shared_mem(_shared_mem),
      entry(_entry) {}

   dim3 grid_dim;
   dim3 block_dim;
   unsigned int shared_mem;
   function_info * entry;

};

typedef enum _device_launch_op_name {
   DEVICE_LAUNCH_AGG, //aggregated blocks
   DEVICE_LAUNCH_CHILD, //new child kernel
   DEVICE_LAUNCH_DCC //Po-Han: dynamic child-thread consolidation
}device_launch_op_name;

//Po-Han: hand-coded application ids for DCC
/*typedef enum _application_id {
  BFS,
  AMR,
  JOIN,
  SSSP,
  COLOR,
  MIS,
  PAGERANK,
  KMEANS
  } application_id;*/

typedef enum _dev_launch_type {
   NORMAL,
   PARENT_FINISHED,
   PARENT_BLOCK_SYNC
} dev_launch_type;

class device_launch_operation_t {

public:
   device_launch_operation_t() {}
   device_launch_operation_t(kernel_info_t *_grid,
     CUstream_st * _stream,
     agg_block_group_t * _agg_block_group,
     device_launch_op_name _op_name) :
      grid(_grid), stream(_stream),
      agg_block_group(_agg_block_group),
      op_name(_op_name) {}

   kernel_info_t * grid; //either a new child grid, or a launched grid

   //For child kernel only
   CUstream_st * stream; 

   //For agg only
   agg_block_group_t * agg_block_group;

   //AGG op or new child kernel
   device_launch_op_name op_name;

};


std::map<void *, device_launch_config_t> g_cuda_device_launch_param_map;
std::list<device_launch_operation_t> g_cuda_device_launch_op;
extern stream_manager *g_stream_manager;
bool g_agg_blocks_support = false;
unsigned long long g_total_param_size = 0;
unsigned long long g_max_total_param_size = 0;
/*Po-Han: dynamic child-thread consolidation support*/
std::list<dcc_kernel_distributor_t> g_cuda_dcc_kernel_distributor;
bool g_dyn_child_thread_consolidation = false;
unsigned g_dyn_child_thread_consolidation_version = 0;
unsigned g_dcc_timeout_threshold = 0;
unsigned pending_child_threads = 0;
signed long long param_buffer_size = 0, param_buffer_usage = 0;
signed int kernel_param_usage = 0;
static const signed per_kernel_param_usage[10] = {12, 16, 16, 12, 0, 8, 8, 16, 12, 12};
static const unsigned per_kernel_optimal_child_size[10] = {16640, 9984, 23296, 9984, -1, 16640, 16640, -1, -1, -1};
//static const unsigned per_kernel_parent_block_cnt[10] = {3, -1, 2, 3, -1, -1, -1, -1, -1, -1};
application_id g_app_name = BFS;
bool g_restrict_parent_block_count = false;
bool param_buffer_full = false;
extern unsigned g_max_param_buffer_size;
extern unsigned g_param_buffer_thres_high;
std::list<int> target_parent_list;
#if 0
std::string bfs_parent_k("bfsCdpExpandKernel");
std::string join_parent_k("joinCdpMainJoinKernel");
std::string sssp_parent_k("ssspCdpExpandKernel");
std::string mis_parent_k1("mis1");
std::string mis_parent_k2("mis2");
std::string pr_parent_k1("inicsr");
std::string pr_parent_k2("spmv_csr_scalar_kernel");
std::string kmeans_parent_k("kmeansPoint");
std::string bc_parent_k1("bfs_kernel");
std::string bc_parent_k2("backtrack_kernel");
#endif
std::string pr_k1("inicsr_CdpKernel");
std::string pr_k2("spmv_csr_scalar_CdpKernel");
std::string bc_k1("bfs_CdpKernel");
std::string bc_k2("backtrack_CdpKernel");
std::string mis_k1("mis1_CdpKernel");
std::string mis_k2("mis2_CdpKernel");

bool compare_dcc_kd_entry(const dcc_kernel_distributor_t &a, const dcc_kernel_distributor_t &b)
{
   return (a.thread_count > b.thread_count);
}

//Handling device runtime api:
//void * cudaGetParameterBufferV2(void *func, dim3 grid_dimension, dim3 block_dimension, unsigned int shared_memSize)
void gpgpusim_cuda_getParameterBufferV2(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func)
{
   /* Po-Han: different treatment if dynamic child-thread consolidation is enabled
    * 1) allocate the parameter buffer in a separate memory space (no timing simulation) for the following write instructions 
    * 2) pre-allocate a kernel_info_t in the kernel distributor (remain invalid since the parameters are not ready)
    */
   DEV_RUNTIME_REPORT("Calling cudaGetParameterBufferV2");

   unsigned n_return = target_func->has_return();
   assert(n_return);
   unsigned n_args = target_func->num_args();
   assert( n_args == 4 );

   function_info * child_kernel_entry;
   struct dim3 grid_dim, block_dim;
   unsigned int shared_mem;

   for( unsigned arg=0; arg < n_args; arg ++ ) {
      const operand_info &actual_param_op = pI->operand_lookup(n_return+1+arg); //param#
      const symbol *formal_param = target_func->get_arg(arg); //cudaGetParameterBufferV2_param_#
      unsigned size=formal_param->get_size_in_bytes();
      assert( formal_param->is_param_local() );
      assert( actual_param_op.is_param_local() );
      addr_t from_addr = actual_param_op.get_symbol()->get_address();

      if(arg == 0) {//function_info* for the child kernel
         unsigned long long buf;
         assert(size == sizeof(function_info *));
         thread->m_local_mem->read(from_addr, size, &buf);
         child_kernel_entry = (function_info *)buf;
         assert(child_kernel_entry);
         DEV_RUNTIME_REPORT("child kernel name " << child_kernel_entry->get_name());
      }
      else if(arg == 1) { //dim3 grid_dim for the child kernel
         assert(size == sizeof(struct dim3));
         thread->m_local_mem->read(from_addr, size, & grid_dim);
         DEV_RUNTIME_REPORT("grid (" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")");
      }
      else if(arg == 2) { //dim3 block_dim for the child kernel
         assert(size == sizeof(struct dim3));
         thread->m_local_mem->read(from_addr, size, & block_dim);
         DEV_RUNTIME_REPORT("block (" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")");
      }
      else if(arg == 3) { //unsigned int shared_mem
         assert(size == sizeof(unsigned int));
         thread->m_local_mem->read(from_addr, size, & shared_mem);
         DEV_RUNTIME_REPORT("shared memory " << shared_mem);
      }
   }

   //get total child kernel argument size and malloc buffer in global memory
   unsigned child_kernel_arg_size = child_kernel_entry->get_args_aligned_size();
   void *param_buffer;
   if( g_dyn_child_thread_consolidation ){
      // Po-Han: DCC implementation, allocate child kernel parameters in another memory space
      param_buffer = thread->get_gpu()->dcc_param_malloc(child_kernel_arg_size);
      DEV_RUNTIME_REPORT("DCC: child kernel arg pre-allocation: size " << child_kernel_arg_size << ", parameter buffer allocated at " << param_buffer);
      //g_total_param_size += child_kernel_arg_size; 
#if 0
      kernel_param_usage = per_kernel_param_usage[g_app_name];
      if(child_kernel_entry->get_name().find(mis_k1) != std::string::npos || (child_kernel_entry->get_name().find(pr_k2) != std::string::npos))
         kernel_param_usage += 8;
      param_buffer_usage += kernel_param_usage;
      if( (float)param_buffer_usage > (float)g_max_param_buffer_size * 0.8){
         param_buffer_full = true;
         DEV_RUNTIME_REPORT("DCC: parameter buffer usage " << param_buffer_usage << ", exceeds 80% (size = " << g_max_param_buffer_size << ")");
      }
      if(param_buffer_usage > param_buffer_size){
         param_buffer_size = param_buffer_usage;
         DEV_RUNTIME_REPORT("DCC: maximum parameter buffer usage = " << param_buffer_size);
      }
#endif
   } else {
      param_buffer = thread->get_gpu()->gpu_malloc(child_kernel_arg_size);
      g_total_param_size += ((child_kernel_arg_size + 255) / 256 * 256);
      DEV_RUNTIME_REPORT("child kernel arg size total " << child_kernel_arg_size << ", parameter buffer allocated at " << param_buffer);
      if(g_total_param_size > g_max_total_param_size)
         g_max_total_param_size = g_total_param_size;
   }

   if( g_dyn_child_thread_consolidation ){
      //create child kernel_info_t and index it with parameter_buffer address

      // store the total thread number of current child kernel
      unsigned int total_thread_count = (unsigned int)(grid_dim.x*grid_dim.y*grid_dim.z*block_dim.x*block_dim.y*block_dim.z);

      // compute optimal block size for child kernel 
      // It's an inefficient implementation since it only needs to be done once per child-kernel
      unsigned int reg_usage = ptx_kernel_nregs(child_kernel_entry);
      unsigned int reg_per_SM = thread->get_gpu()->num_registers_per_core();
      unsigned int max_threads_per_SM = reg_per_SM / reg_usage;
      max_threads_per_SM = gs_min2(max_threads_per_SM, thread->get_gpu()->threads_per_core());
      unsigned int min_threads_per_block = max_threads_per_SM / thread->get_gpu()->max_cta_per_core();
      unsigned int optimal_threads_per_block;
      for( optimal_threads_per_block = min_threads_per_block; optimal_threads_per_block <= max_threads_per_SM; optimal_threads_per_block++ )
         if( optimal_threads_per_block % 32 == 0 ) 
            break;
      grid_dim.x = (total_thread_count + optimal_threads_per_block - 1) / optimal_threads_per_block;
      block_dim.x = optimal_threads_per_block;
      grid_dim.y = grid_dim.z = block_dim.y = block_dim.z = 1;
      // compute optimal threads per kernel
      unsigned int max_con_kernels = thread->get_gpu()->get_config().get_max_concurrent_kernel();
      unsigned int num_shaders = thread->get_gpu()->get_config().num_shader();
      unsigned int optimal_threads_per_kernel = ((num_shaders * (max_threads_per_SM / optimal_threads_per_block) + max_con_kernels) / max_con_kernels) * optimal_threads_per_block ;

      DEV_RUNTIME_REPORT("DCC: child kernel properties -- #thread " << total_thread_count << ", reg/thread " << reg_usage << ", max_thread/SM " << max_threads_per_SM << ", optimal_block_size " << optimal_threads_per_block << ", optimal_kernel_size " << optimal_threads_per_kernel);

      // pre-allocate child kernel entry and link it with parent kernel
      kernel_info_t * device_grid = new kernel_info_t(grid_dim, block_dim, child_kernel_entry);
      device_grid->launch_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      device_grid->is_child = true;
      device_grid->m_param_mem_map[total_thread_count] = device_grid->get_param_memory(-1); //Po-Han DCC: setting to the kernel paramenter map
      kernel_info_t & parent_grid = thread->get_kernel();
      device_grid->add_parent(&parent_grid, thread);  
      DEV_RUNTIME_REPORT("DCC: preallocate child kernel at kernel distributor by " << parent_grid.name() << ", cta (" <<
        thread->get_ctaid().x << ", " << thread->get_ctaid().y << ", " << thread->get_ctaid().z <<
        "), thread (" << thread->get_tid().x << ", " << thread->get_tid().y << ", " << thread->get_tid().z << ")");
      
#if 0
      /* modify target_parent_list to block the execution of parent warps if necessary */
      if(param_buffer_full){
         if(std::find(target_parent_list.begin(), target_parent_list.end(), parent_grid.get_uid()) == target_parent_list.end()){
            target_parent_list.push_back(parent_grid.get_uid());
            DEV_RUNTIME_REPORT("DCC: add parent kernel id = " << parent_grid.get_uid() << "to target parent list.");
         }
      } else {
         if(std::find(target_parent_list.begin(), target_parent_list.end(), parent_grid.get_uid()) != target_parent_list.end()){
            target_parent_list.remove(parent_grid.get_uid());
            DEV_RUNTIME_REPORT("DCC: remove parent kernel id = " << parent_grid.get_uid() << "to target parent list.");
         }
      }
#endif
      // initialize the kernel distributor entry
      dcc_kernel_distributor_t distributor_entry(device_grid, total_thread_count, optimal_threads_per_block, optimal_threads_per_kernel, param_buffer, thread->get_agg_group_id(), thread->get_ctaid());
      g_cuda_dcc_kernel_distributor.push_back(distributor_entry);
      DEV_RUNTIME_REPORT("DCC: kernel distributor with size " << g_cuda_dcc_kernel_distributor.size());
   }

   //store param buffer address and launch config
   device_launch_config_t device_launch_config(grid_dim, block_dim, shared_mem, child_kernel_entry);
   assert(g_cuda_device_launch_param_map.find(param_buffer) == g_cuda_device_launch_param_map.end());
   g_cuda_device_launch_param_map[param_buffer] = device_launch_config;


   //copy the buffer address to retval0
   const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
   const symbol *formal_return = target_func->get_return_var(); //void *
   unsigned int return_size = formal_return->get_size_in_bytes();
   DEV_RUNTIME_REPORT("cudaGetParameterBufferV2 return value has size of " << return_size);
   assert(actual_return_op.is_param_local());
   assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size && return_size == sizeof(void *));
   addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
   thread->m_local_mem->write(ret_param_addr, return_size, &param_buffer, NULL, NULL);


}

bool merge_two_kernel_distributor_entry(dcc_kernel_distributor_t *kd_entry_1, dcc_kernel_distributor_t *kd_entry_2, bool ForceMerge, int target_size, bool &remaining)
{
   bool continous_offset = false;
   memory_space *mspace1, *mspace2, *new_mspace;
   std::map<unsigned int, memory_space *>::iterator it;
   it = kd_entry_1->kernel_grid->m_param_mem_map.begin();
   mspace1 = it->second;
   it = kd_entry_2->kernel_grid->m_param_mem_map.begin();
   mspace2 = it->second;
   unsigned int total_thread_1, total_thread_2;
   int offset_a_1, offset_a_2, offset_b_1, offset_b_2;
   unsigned int total_thread_sum, total_thread_offset;
   unsigned int kernel_param_size;
   int new_offset_b_1, new_offset_b_2;
   unsigned int num_blocks, thread_per_block;
   unsigned int stride_1, stride_2;
   dim3 gDim;
   unsigned parent_block_idx_1, parent_block_idx_2;
   size_t found1, found2;

   remaining = false;
   switch(g_app_name){
   case BFS:
      //[offset_a (4B), total_thread (4B), base_a (8B), base_b (8B), offset_b (4B)]
      mspace1->read((size_t) 0, 4, &offset_a_1);
      mspace1->read((size_t) 4, 4, &total_thread_1);
      mspace1->read((size_t)24, 4, &offset_b_1);
      mspace2->read((size_t) 0, 4, &offset_a_2);
      mspace2->read((size_t) 4, 4, &total_thread_2);
      mspace2->read((size_t)24, 4, &offset_b_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         DEV_RUNTIME_REPORT("DCC: BFS continous -> child1 (" << offset_a_1 << ", " << offset_b_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << offset_b_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      } 
      total_thread_offset = 4;
      kernel_param_size = 28;
      break;
   case AMR:
      //[total_thread (4B), base_a (8B), base_b (8B), var_a (F4B), const_a (4B), const_b (4B), const_c (F4B), offset (4B), var_b (4B), const (8B)]
      mspace1->read((size_t) 0, 4, &total_thread_1);
      mspace2->read((size_t) 0, 4, &total_thread_2);
      mspace1->read((size_t)40, 4, &offset_a_1);
      mspace2->read((size_t)40, 4, &offset_a_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         DEV_RUNTIME_REPORT("DCC: AMR continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      }
      total_thread_offset = 0;
      kernel_param_size = 56;
      break;
   case JOIN:
      //[base_a (8B), base_b (8B), offset_a (4B), total_thread (4B), offset_b (4B), var (4B)]
      mspace1->read((size_t)16, 4, &offset_a_1);
      mspace2->read((size_t)16, 4, &offset_a_2);
      mspace1->read((size_t)20, 4, &total_thread_1);
      mspace2->read((size_t)20, 4, &total_thread_2);
      mspace1->read((size_t)24, 4, &offset_b_1);
      mspace2->read((size_t)24, 4, &offset_b_2);
      
//      assert( kd_entry_1->thread_count == total_thread_1 );
//      assert( kd_entry_2->thread_count == total_thread_2 );

      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         DEV_RUNTIME_REPORT("DCC: JOIN continous -> child1 (" << offset_a_1 << ", " << offset_b_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << offset_b_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      }
      total_thread_offset = 20;
      kernel_param_size = 32;
      break;
   case SSSP:
      //[offset (4B), total_thread (4B), var (4B), base_a~d (8Bx4), const_a (8B), const_b (4B)]
      mspace1->read((size_t) 0, 4, &offset_a_1);
      mspace2->read((size_t) 0, 4, &offset_a_2);
      mspace1->read((size_t) 4, 4, &total_thread_1);
      mspace2->read((size_t) 4, 4, &total_thread_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         DEV_RUNTIME_REPORT("DCC: SSSP continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      }
      total_thread_offset = 4;
      kernel_param_size = 60;
      break;
   case COLOR:
      //[offset (4B), total_thread (4B), base_a~d (8Bx3), var (8B)]
      mspace1->read((size_t) 0, 4, &offset_a_1);
      mspace2->read((size_t) 0, 4, &offset_a_2);
      mspace1->read((size_t) 4, 4, &total_thread_1);
      mspace2->read((size_t) 4, 4, &total_thread_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      parent_block_idx_1 = kd_entry_1->kernel_grid->m_parent_threads.front()->get_block_idx();
      parent_block_idx_2 = kd_entry_2->kernel_grid->m_parent_threads.front()->get_block_idx();
      if(parent_block_idx_1 != parent_block_idx_2){
         return false;
      }
      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         DEV_RUNTIME_REPORT("DCC: COLOR continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      }
      total_thread_offset = 4;
      kernel_param_size = 48;
      break;
   case MIS:
      found1 = kd_entry_1->kernel_grid->name().find(mis_k1);
      found2 = kd_entry_1->kernel_grid->name().find(mis_k2);
      //[offset (4B), total_thread (4B), var (8B), base_a~c (8Bx3)]
      mspace1->read((size_t) 0, 4, &offset_a_1);
      mspace2->read((size_t) 0, 4, &offset_a_2);
      mspace1->read((size_t) 4, 4, &total_thread_1);
      mspace2->read((size_t) 4, 4, &total_thread_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

/*      parent_block_idx_1 = kd_entry_1->kernel_grid->m_parent_threads.front()->get_block_idx();
      parent_block_idx_2 = kd_entry_2->kernel_grid->m_parent_threads.front()->get_block_idx();
      if(parent_block_idx_1 != parent_block_idx_2){
         return false;
      }*/
      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         if(found1 != std::string::npos){ //kernel mis1
            DEV_RUNTIME_REPORT("DCC: MIS1 continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         } else if(found2 != std::string::npos){ //kernel mis2
            DEV_RUNTIME_REPORT("DCC: MIS2 continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         }
         continous_offset = true;
      }
      total_thread_offset = 4;
      if(found1 != std::string::npos){ //kernel mis1
         kernel_param_size = 40;
      } else if(found2 != std::string::npos){ //kernel mis2
         kernel_param_size = 32;
      }
      break;
   case PAGERANK:
      found1 = kd_entry_1->kernel_grid->name().find(pr_k1);
      found2 = kd_entry_1->kernel_grid->name().find(pr_k2);
      //[offset (4B), total_thread (4B), var (8B), base_a~c (8Bx3)]
      mspace1->read((size_t) 0, 4, &offset_a_1);
      mspace2->read((size_t) 0, 4, &offset_a_2);
      mspace1->read((size_t) 4, 4, &total_thread_1);
      mspace2->read((size_t) 4, 4, &total_thread_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      parent_block_idx_1 = kd_entry_1->kernel_grid->m_parent_threads.front()->get_block_idx();
      parent_block_idx_2 = kd_entry_2->kernel_grid->m_parent_threads.front()->get_block_idx();
      
      if(found1 != std::string::npos){ //kernel inicsr
         //[offset (4B), total_thread (4B), base_a~c (8Bx3)]
         if( offset_a_1 + total_thread_1 == offset_a_2 ){
            DEV_RUNTIME_REPORT("DCC: PAGERANK-" << pr_k1 << " continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
            continous_offset = true;
         }
         total_thread_offset = 4;
         kernel_param_size = 32;
      }else if(found2 != std::string::npos){ //kernel spmv_csr_scalar
/*         if(parent_block_idx_1 != parent_block_idx_2){
            return false;
         }*/
         //[offset (4B), total_thread (4B), var (8B), base_a~c (8Bx3)]
         if( offset_a_1 + total_thread_1 == offset_a_2 ){
            DEV_RUNTIME_REPORT("DCC: PAGERANK-" << pr_k2 << " continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
            continous_offset = true;
         }
         total_thread_offset = 4;
         kernel_param_size = 40;
      }else{
         DEV_RUNTIME_REPORT("DCC: unsupported pagerank child kernel name");
         assert(0);
      }
      break;
   case KMEANS:
      //[total_thread (4B), offset_2 (4B), stride (4B), base_a~b (8Bx2), offset_1 (4B), var (8B)]
      mspace1->read((size_t)32, 4, &offset_a_1);
      mspace2->read((size_t)32, 4, &offset_a_2);
      mspace1->read((size_t) 0, 4, &total_thread_1);
      mspace2->read((size_t) 0, 4, &total_thread_2);
      mspace1->read((size_t) 4, 4, &offset_b_1);
      mspace2->read((size_t) 4, 4, &offset_b_2);
      mspace1->read((size_t) 8, 4, &stride_1);
      mspace2->read((size_t) 8, 4, &stride_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );
      assert( stride_1 == stride_2 );

      parent_block_idx_1 = kd_entry_1->kernel_grid->m_parent_threads.front()->get_block_idx();
      parent_block_idx_2 = kd_entry_2->kernel_grid->m_parent_threads.front()->get_block_idx();
      if(parent_block_idx_1 != parent_block_idx_2){
         return false;
      }
      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         DEV_RUNTIME_REPORT("DCC: KMEANS continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      }
      total_thread_offset = 0;
      kernel_param_size = 48;
      break;
   case BFS_RODINIA:
      //[var (4B), total_thread (4B), offset (4B), base_a~d (8Bx4)]
      mspace1->read((size_t) 8, 4, &offset_a_1);
      mspace2->read((size_t) 8, 4, &offset_a_2);
      mspace1->read((size_t) 4, 4, &total_thread_1);
      mspace2->read((size_t) 4, 4, &total_thread_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         //child kernel 2 can be catenated after child kernel 1
         DEV_RUNTIME_REPORT("DCC: BFS_RODINIA continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      }
      total_thread_offset = 4;
      kernel_param_size = 44;
      break;
   case BC:
      found1 = kd_entry_1->kernel_grid->name().find(bc_k1);
      found2 = kd_entry_1->kernel_grid->name().find(bc_k2);
      //[offset (4B), total_thread (4B), base_a~d (8Bx4), var (4B), const (4B)]
      mspace1->read((size_t) 0, 4, &offset_a_1);
      mspace2->read((size_t) 0, 4, &offset_a_2);
      mspace1->read((size_t) 4, 4, &total_thread_1);
      mspace2->read((size_t) 4, 4, &total_thread_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      parent_block_idx_1 = kd_entry_1->kernel_grid->m_parent_threads.front()->get_block_idx();
      parent_block_idx_2 = kd_entry_2->kernel_grid->m_parent_threads.front()->get_block_idx();
      if(parent_block_idx_1 != parent_block_idx_2){
         return false;
      }
      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         if(found1 != std::string::npos){ //kernel inicsr
            DEV_RUNTIME_REPORT("DCC: BC-bfs continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         } else if(found1 != std::string::npos){ //kernel inicsr
            DEV_RUNTIME_REPORT("DCC: BC-backtrack continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         }
         continous_offset = true;
      }
      if(found1 != std::string::npos){ //kernel bfs
         kernel_param_size = 48;
      } else if(found1 != std::string::npos){ //kernel backtrack
         kernel_param_size = 48;
      }
      total_thread_offset = 4;
      break;
   default:
      DEV_RUNTIME_REPORT("Dynamic Child-thread Consolidation: unsupported application");
      assert(0);
      break;
   }

   if(target_size != -1 && total_thread_1 > target_size) return false; //enough thread for a child kernel, merge is unnecessary

   unsigned remaining_count = 0;

   if(continous_offset || ForceMerge){
      //child kernel 2 can be catenated after child kernel 1

      // adjust thread count
      total_thread_sum = total_thread_1 + total_thread_2;
      if( (target_size != -1) && (total_thread_sum > target_size) ){
//         if (kd_entry_2->kernel_grid->m_param_mem_map.size() > 1) //if the latter kd entry is composed of more than 1 child kernel, find another one
//            return false;
         remaining = true;
         remaining_count = total_thread_sum - target_size; //record the number of threads that should be cut-off
         new_mspace = new memory_space_impl<256>("param", 256);
         //copy the kernel parameters of the splitting kernel into a new memory space
//         it = kd_entry_2->kernel_grid->m_param_mem_map.begin();
//         for(unsigned n = 0; n < kernel_param_size; n += 4) {
//            unsigned int oneword;
//            it->second->read((size_t) n, 4, &oneword);
//            new_mspace->write((size_t) n, 4, &oneword, NULL, NULL); 
//         }
      }
      total_thread_sum -= remaining_count;
      kd_entry_1->thread_count = total_thread_sum;
      mspace1->write((size_t)total_thread_offset, 4, &total_thread_sum, NULL, NULL);
      if(remaining){
         kd_entry_2->thread_count = remaining_count;
         gDim = kd_entry_2->kernel_grid->get_grid_dim(-1);
         thread_per_block = kd_entry_1->kernel_grid->threads_per_cta();
         num_blocks = (remaining_count + thread_per_block - 1) / thread_per_block;
         gDim.x = num_blocks;
         kd_entry_2->kernel_grid->set_grid_dim(gDim);
         DEV_RUNTIME_REPORT("Merge kernel " << kd_entry_2->kernel_grid->get_uid() << "(" << total_thread_2 << " threads) into kernel " << kd_entry_1->kernel_grid->get_uid() << "(" << total_thread_1 << " threads), target kernel size " << target_size << ", kernel "<< kd_entry_2->kernel_grid->get_uid() << " has " << remaining_count << " threads remaining.");
      }
      

      // adjust grid dimension
      gDim = kd_entry_1->kernel_grid->get_grid_dim(-1);
      thread_per_block = kd_entry_1->kernel_grid->threads_per_cta();
      num_blocks = (total_thread_sum + thread_per_block - 1) / thread_per_block;
      gDim.x = num_blocks;
      kd_entry_1->kernel_grid->set_grid_dim(gDim);

      // set up launch cycle
      kd_entry_1->kernel_grid->launch_cycle = gs_min2(kd_entry_1->kernel_grid->launch_cycle, kd_entry_2->kernel_grid->launch_cycle);
//      DEV_RUNTIME_REPORT("DCC: merge child kernel " << kd_entry_2->kernel_grid->get_uid() << " into child kernel " << kd_entry_1->kernel_grid->get_uid() << ", new threads " << total_thread_sum << ", new blocks " << num_blocks);

      // merge parameter buffer
      bool boundary = false, split = false;
      unsigned offset;
      unsigned split_size = 0;
      std::map<unsigned int, memory_space *> new_map;
      for( it = kd_entry_2->kernel_grid->m_param_mem_map.begin(); it != kd_entry_2->kernel_grid->m_param_mem_map.end(); it++ ){
         if (!split){
            offset = it->first + total_thread_1;
            DEV_RUNTIME_REPORT("DCC pre-split: copy kernel param " << it->second << " old offset " << it->first << " new offset " << offset );
            if( target_size != -1 ){
               if(offset > target_size) { //boundary parameter buffer -> duplicate
                  boundary = true;
                  for(unsigned n = 0; n < kernel_param_size; n += 4) {
                     unsigned int oneword;
                     it->second->read((size_t) n, 4, &oneword);
                     new_mspace->write((size_t) n, 4, &oneword, NULL, NULL); 
                  }
                  split_size = offset - target_size;
                  DEV_RUNTIME_REPORT("DCC pre-split: found boundary, set new param mem offset at " << split_size);
                  new_mspace->write((size_t)total_thread_offset, 4, &remaining_count, NULL, NULL);
                  new_map[split_size] = new_mspace;
               }else if(offset == target_size){ //exact boundary
                  DEV_RUNTIME_REPORT("DCC pre-split: found exact boundary");
                  split = true;
               }
            }
//            DEV_RUNTIME_REPORT("DCC pre-split: copy kernel param " << it->second << " old offset " << it->first << " new offset " << offset );
            it->second->write((size_t)total_thread_offset, 4, &total_thread_sum, NULL, NULL);
            kd_entry_1->kernel_grid->m_param_mem_map[offset] = it->second;
         } else { // splitted -> modifying parameter memory maps
            DEV_RUNTIME_REPORT("DCC post-split: copy kernel param " << it->second << " old offset " << it->first << " new offset " << (it->first-(total_thread_2 - remaining_count)) );
            it->second->write((size_t)total_thread_offset, 4, &remaining_count, NULL, NULL);
            new_map[it->first-(total_thread_2 - remaining_count)] = it->second;
         }

         switch(g_app_name){
         case BFS:
            it->second->read((size_t) 0, 4, &offset_b_1);
            it->second->read((size_t)24, 4, &offset_b_2);
            if(remaining && !split && boundary ){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_offset_b_2 = offset_b_2 + total_thread_2 - remaining_count;
               new_mspace->write((size_t) 0, 4, &new_offset_b_1, NULL, NULL);
               new_mspace->write((size_t)24, 4, &new_offset_b_2, NULL, NULL);
            }
            if(!split){
               offset_b_1 -= total_thread_1;
               offset_b_2 -= total_thread_1;
            } else {
               offset_b_1 += (total_thread_2 - remaining_count);
               offset_b_2 += (total_thread_2 - remaining_count);
            }
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            it->second->write((size_t)24, 4, &offset_b_2, NULL, NULL);
            break;
         case AMR:
            it->second->read((size_t)40, 4, &offset_b_1);
            if(remaining && !split && boundary){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_mspace->write((size_t)40, 4, &new_offset_b_1, NULL, NULL);
            }
            if(!split){
               offset_b_1 -= total_thread_1;
            }else{
               offset_b_1 += (total_thread_2 - remaining_count);
            }
            it->second->write((size_t)40, 4, &offset_b_1, NULL, NULL);
            break;
         case JOIN:
            it->second->read((size_t)16, 4, &offset_b_1);
            it->second->read((size_t)24, 4, &offset_b_2);
            if(remaining && !split && boundary){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_offset_b_2 = offset_b_2 + total_thread_2 - remaining_count;
               new_mspace->write((size_t)16, 4, &new_offset_b_1, NULL, NULL);
               new_mspace->write((size_t)24, 4, &new_offset_b_2, NULL, NULL);
            }
            if(!split){
               offset_b_1 -= total_thread_1;
               offset_b_2 -= total_thread_1;
            } else {
               offset_b_1 += (total_thread_2 - remaining_count);
               offset_b_2 += (total_thread_2 - remaining_count);
            }
            it->second->write((size_t)16, 4, &offset_b_1, NULL, NULL);
            it->second->write((size_t)24, 4, &offset_b_2, NULL, NULL);
            break;
         case SSSP:
            it->second->read((size_t) 0, 4, &offset_b_1);
            if(remaining && !split && boundary){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_mspace->write((size_t) 0, 4, &new_offset_b_1, NULL, NULL);
            }
            if(!split){
               offset_b_1 -= total_thread_1;
            }else{
               offset_b_1 += (total_thread_2 - remaining_count);
            }
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            break;
         case COLOR:
            assert(!remaining && !split && !boundary);
            it->second->read((size_t) 0, 4, &offset_b_1);
            offset_b_1 -= total_thread_1;
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            break;
         case MIS:
//            if(found1 != std::string::npos) assert(!remaining && !split && !boundary);
            it->second->read((size_t) 0, 4, &offset_b_1);
//            if(found2 != std::string::npos){
               if(remaining && !split && boundary){
                  new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
                  new_mspace->write((size_t) 0, 4, &new_offset_b_1, NULL, NULL);
               }
               if(!split){
                  offset_b_1 -= total_thread_1;
               }else{
                  offset_b_1 += (total_thread_2 - remaining_count);
               }
//            } else {
//               offset_b_1 -= total_thread_1;
//            }
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            break;
         case PAGERANK:
            /*if(found2 != std::string::npos) assert(!remaining && !split && !boundary); */
            it->second->read((size_t) 0, 4, &offset_b_1);
//            if(found1 != std::string::npos){
               if(remaining && !split && boundary){
                  new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
                  new_mspace->write((size_t) 0, 4, &new_offset_b_1, NULL, NULL);
               }
               if(!split){
                  offset_b_1 -= total_thread_1;
               }else{
                  offset_b_1 += (total_thread_2 - remaining_count);
               }
//            } else {
//               offset_b_1 -= total_thread_1;
//            }
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            break;
         case KMEANS:
            assert(!remaining && !split && !boundary);
            it->second->read((size_t)32, 4, &offset_b_1);
            offset_b_1 -= total_thread_1;
            it->second->write((size_t)32, 4, &offset_b_1, NULL, NULL);
            it->second->read((size_t) 4, 4, &offset_b_2);
            offset_b_2 -= total_thread_1 * stride_1;
            it->second->write((size_t) 4, 4, &offset_b_2, NULL, NULL);
            break;
         case BFS_RODINIA:
            it->second->read((size_t) 8, 4, &offset_b_1);
            if(remaining && !split && boundary){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_mspace->write((size_t) 8, 4, &new_offset_b_1, NULL, NULL);
            }
            if(!split){
               offset_b_1 -= total_thread_1;
            }else{
               offset_b_1 += (total_thread_2 - remaining_count);
            }
            it->second->write((size_t) 8, 4, &offset_b_1, NULL, NULL);
            break;
         case BC:
            assert(!remaining && !split && !boundary);
            it->second->read((size_t) 0, 4, &offset_b_1);
            offset_b_1 -= total_thread_1;
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            break;
         default:
            DEV_RUNTIME_REPORT("Dynamic Child-thread Consolidation: unsupported application");
            assert(0);
            break;
         }

         if(!split){
//            DEV_RUNTIME_REPORT("DCC pre-split: copy kernel param " << it->second << " old offset " << it->first << " new offset " << offset );
            //unsigned tmp_key = it->first;
            //it--;
            //kd_entry_2->kernel_grid->m_param_mem_map.erase(tmp_key);
            if(boundary){
               split = true;
            }
         }
      }

      if(split && remaining){
         //new_map[split_size] = new_mspace;
         kd_entry_2->kernel_grid->m_param_mem_map.clear();
         for(it=new_map.begin(); it!=new_map.end(); it++){
            kd_entry_2->kernel_grid->m_param_mem_map[it->first] = it->second;
            DEV_RUNTIME_REPORT("DCC post-split: copy param mem map at offset " << it->first << " back to kernel " << kd_entry_2->kernel_grid->get_uid());
         }
         kd_entry_2->kernel_grid->set_param_mem(new_mspace);
      }

      /*/ set new kernel as candidate
        if ( (total_thread_sum % thread_per_block == 0) || ( (total_thread_sum % thread_per_block) > (unsigned int)(0.9 * thread_per_block)  )){
        kd_entry_1->candidate = true;
        DEV_RUNTIME_REPORT("DCC: set child kernel " << kd_entry_1->kernel_grid << " as candidate");
        } else {
        kd_entry_1->candidate = false;
        }*/

      // set up parent thread and merge count
      if(!remaining){
         while(!(kd_entry_2->kernel_grid->m_parent_threads.empty())){
            kd_entry_1->kernel_grid->add_parent(kd_entry_2->kernel_grid->get_parent(), kd_entry_2->kernel_grid->m_parent_threads.front());
            kd_entry_2->kernel_grid->m_parent_threads.pop_front();
         }
         kd_entry_2->kernel_grid->get_parent()->remove_child(kd_entry_2->kernel_grid);
      } else {
         std::list<ptx_thread_info *>::iterator mpt_it;
         for(mpt_it = kd_entry_2->kernel_grid->m_parent_threads.begin(); mpt_it != kd_entry_2->kernel_grid->m_parent_threads.end(); mpt_it++){
            kd_entry_1->kernel_grid->add_parent(kd_entry_2->kernel_grid->get_parent(), *mpt_it);
         }
         // reset the parameter map of the second kd entry and linked it with new memory space
//         kd_entry_2->kernel_grid->m_param_mem_map.clear();
//         kd_entry_2->kernel_grid->m_param_mem_map[remaining_count] = new_mspace;
//         kd_entry_2->kernel_grid->set_param_mem(new_mspace);
      }
      DEV_RUNTIME_REPORT("DCC: child kernel " << kd_entry_2->kernel_grid->get_uid() << " merged into child kernel " << kd_entry_1->kernel_grid->get_uid() << ", new threads " << total_thread_sum << ", new blocks " << num_blocks << ", kernel " << kd_entry_1->kernel_grid->get_parent()->get_uid() << " now has " << kd_entry_1->kernel_grid->get_parent()->get_child_count() << " child kernels.");
      kd_entry_1->merge_count += kd_entry_2->merge_count;

      return true;
   }

   return false;
}


//Handling device runtime api:
//cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
void gpgpusim_cuda_launchDeviceV2(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func) {
   DEV_RUNTIME_REPORT("Calling cudaLaunchDeviceV2");

   unsigned n_return = target_func->has_return();
   assert(n_return);
   unsigned n_args = target_func->num_args();
   assert( n_args == 2 );

   kernel_info_t * device_grid = NULL;
   function_info * device_kernel_entry = NULL;
   void * parameter_buffer;
   struct CUstream_st * child_stream;
   device_launch_config_t config;
   device_launch_operation_t device_launch_op;
   dcc_kernel_distributor_t *k_dis;

   for( unsigned arg=0; arg < n_args; arg ++ ) {
      const operand_info &actual_param_op = pI->operand_lookup(n_return+1+arg); //param#
      const symbol *formal_param = target_func->get_arg(arg); //cudaLaunchDeviceV2_param_#
      unsigned size=formal_param->get_size_in_bytes();
      assert( formal_param->is_param_local() );
      assert( actual_param_op.is_param_local() );
      addr_t from_addr = actual_param_op.get_symbol()->get_address();

      if(arg == 0) {//parameter buffer for child kernel (in global memory)
         //get parameter_buffer from the cudaLaunchDeviceV2_param0
         assert(size == sizeof(void *));
         thread->m_local_mem->read(from_addr, size, &parameter_buffer);
         if( g_dyn_child_thread_consolidation ) {
            assert((size_t)parameter_buffer >= DCC_PARAM_START);
            DEV_RUNTIME_REPORT("Parameter buffer locating at on-chip kernel parameter memory " << parameter_buffer);
         } else {
            assert((size_t)parameter_buffer >= GLOBAL_HEAP_START);
            DEV_RUNTIME_REPORT("Parameter buffer locating at global memory " << parameter_buffer);
         }

         //get either child grid or native grid info through parameter_buffer address
         assert(g_cuda_device_launch_param_map.find(parameter_buffer) != g_cuda_device_launch_param_map.end());
         config = g_cuda_device_launch_param_map[parameter_buffer];
         //device_grid = op.grid;
         device_kernel_entry = config.entry;
         DEV_RUNTIME_REPORT("find device kernel " << device_kernel_entry->get_name());

         //copy data in parameter_buffer to device kernel param memory
         unsigned device_kernel_arg_size = device_kernel_entry->get_args_aligned_size();
         DEV_RUNTIME_REPORT("device_kernel_arg_size " << device_kernel_arg_size);

         memory_space *device_kernel_param_mem;
         if( g_dyn_child_thread_consolidation ){
            /* Po-Han DCC: 
             * 1) Change the valid bit of the corresponding kernel distributor entry so that it is open for merge
             * 2) Copy kernel parameters to corresponding places
             */
            std::list<dcc_kernel_distributor_t>::iterator kd_entry;
            unsigned int i;
            for(kd_entry = g_cuda_dcc_kernel_distributor.begin(), i = 0;
              kd_entry != g_cuda_dcc_kernel_distributor.end();
              kd_entry++, i++){
               if( kd_entry->parameter_buffer == parameter_buffer ){
                  kd_entry->valid = true;
                  device_grid = kd_entry->kernel_grid; //get kernel descriptor
                  device_kernel_param_mem = kd_entry->kernel_grid->get_param_memory(-1); //get paramenter buffer
                  pending_child_threads += kd_entry->thread_count; //record pending child threads
                  k_dis = &(*kd_entry);
                  /* Parent-child dependency */
                  thread->get_kernel().block_state[thread->get_block_idx()].thread.reset(thread->get_thread_idx());
                  DEV_RUNTIME_REPORT("DCC: activate kernel distributor entry " << i << " with parameter buffer address " << parameter_buffer << ", kernel distributor now has " << pending_child_threads << " pending threads.");
                  DEV_RUNTIME_REPORT("Reset block state for block " << thread->get_block_idx() << " thread " << thread->get_thread_idx());
                  break;
               }
            }
            kernel_param_usage = per_kernel_param_usage[g_app_name];
            if(kd_entry->kernel_grid->name().find(mis_k1) != std::string::npos || (kd_entry->kernel_grid->name().find(pr_k2) != std::string::npos))
               kernel_param_usage += 8;
            param_buffer_usage += kernel_param_usage;
            DEV_RUNTIME_REPORT("DCC: current parameter buffer usage = " << param_buffer_usage);
            if( param_buffer_usage * 100 > g_max_param_buffer_size * g_param_buffer_thres_high){
               if(g_app_name == BFS || g_app_name == AMR || g_app_name == JOIN || g_app_name == SSSP || g_app_name == BFS_RODINIA || g_app_name == PAGERANK || 
                 (g_app_name == MIS /*&& kd_entry->kernel_grid->name().find(mis_k2) != std::string::npos*/) ) {
                  param_buffer_full = true;
               }
               DEV_RUNTIME_REPORT("DCC: parameter buffer usage exceeds " << g_param_buffer_thres_high << "% (size = " << g_max_param_buffer_size << ")");
            }
            if(param_buffer_usage > param_buffer_size){
               param_buffer_size = param_buffer_usage;
               DEV_RUNTIME_REPORT("DCC: maximum parameter buffer usage = " << param_buffer_size);
            }
            /* modify target_parent_list to block the execution of parent warps if necessary */
            kernel_info_t & parent_grid = thread->get_kernel();
            if(param_buffer_full){
               if(std::find(target_parent_list.begin(), target_parent_list.end(), parent_grid.get_uid()) == target_parent_list.end()){
                  target_parent_list.push_back(parent_grid.get_uid());
                  DEV_RUNTIME_REPORT("DCC: add parent kernel " << parent_grid.get_uid() << " to target parent list.");
               }
            } else {
               if(std::find(target_parent_list.begin(), target_parent_list.end(), parent_grid.get_uid()) != target_parent_list.end()){
                  target_parent_list.remove(parent_grid.get_uid());
                  DEV_RUNTIME_REPORT("DCC: remove parent kernel " << parent_grid.get_uid() << " to target parent list.");
               }
            }

         } else {
            //find if the same kernel has been launched before
            device_grid = find_launched_grid(device_kernel_entry);

            if(device_grid == NULL) { //first time launch, as child kernel

               //create child kernel_info_t and index it with parameter_buffer address
               device_grid = new kernel_info_t(config.grid_dim, config.block_dim, device_kernel_entry);
               device_grid->launch_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
               kernel_info_t & parent_grid = thread->get_kernel();
               DEV_RUNTIME_REPORT("child kernel launched by " << parent_grid.name() << ", agg_group_id " <<
                 thread->get_agg_group_id() << ", cta (" <<
                 thread->get_ctaid().x << ", " << thread->get_ctaid().y << ", " << thread->get_ctaid().z <<
                 "), thread (" << thread->get_tid().x << ", " << thread->get_tid().y << ", " << thread->get_tid().z <<
                 ")");
               device_grid->set_parent(&parent_grid, thread->get_agg_group_id(), thread->get_ctaid(), thread->get_tid(), thread->get_block_idx(), thread->get_thread_idx());  
               device_launch_op = device_launch_operation_t(device_grid, NULL, NULL, DEVICE_LAUNCH_CHILD);
               device_kernel_param_mem = device_grid->get_param_memory(-1); //native kernel param
               thread->get_kernel().block_state[thread->get_block_idx()].thread.reset(thread->get_thread_idx());
               DEV_RUNTIME_REPORT("Reset block state for block " << thread->get_block_idx() << " thread " << thread->get_thread_idx());
            }
            else { //launched before, as aggregated blocks
               agg_block_group_t * agg_block_group = new agg_block_group_t(config.grid_dim, config.block_dim, device_grid);

               //add aggregated blocks
               DEV_RUNTIME_REPORT("found launched grid with the same function " << device_grid->get_uid() << 
                 ", appended as aggregated blocks by " << thread->get_kernel().name() << ", agg_group_id " <<
                 thread->get_agg_group_id() << ", cta (" <<
                 thread->get_ctaid().x << ", " << thread->get_ctaid().y << ", " << thread->get_ctaid().z <<
                 "), thread (" << thread->get_tid().x << ", " << thread->get_tid().y << ", " << thread->get_tid().z <<
                 ")");

               device_launch_op = device_launch_operation_t(device_grid, NULL, agg_block_group, DEVICE_LAUNCH_AGG);
               device_kernel_param_mem = agg_block_group->get_param_memory();
            }
         }

         size_t param_start_address = 0;
         //copy in word
         for(unsigned n = 0; n < device_kernel_arg_size; n += 4) {
            unsigned int oneword;
            thread->get_gpu()->get_global_memory()->read((size_t)parameter_buffer + n, 4, &oneword);
            device_kernel_param_mem->write(param_start_address + n, 4, &oneword, NULL, NULL); 
         }

      }
      else if(arg == 1) { //cudaStream for the child kernel
         if(!g_dyn_child_thread_consolidation){
            if(device_launch_op.op_name == DEVICE_LAUNCH_CHILD) {
               assert(size == sizeof(cudaStream_t));
               thread->m_local_mem->read(from_addr, size, &child_stream);

               kernel_info_t & parent_kernel = thread->get_kernel();
               if(child_stream == 0) { //default stream on device for current CTA
                  child_stream = parent_kernel.get_default_stream_cta(thread->get_agg_group_id(), thread->get_ctaid()); 
                  DEV_RUNTIME_REPORT("launching child kernel " << device_grid->get_uid() << 
                    " to default stream of the cta " << child_stream->get_uid() << ": " << child_stream);
               }
               else {
                  assert(parent_kernel.cta_has_stream(thread->get_agg_group_id(), thread->get_ctaid(), child_stream)); 
                  DEV_RUNTIME_REPORT("launching child kernel " << device_grid->get_uid() << 
                    " to stream " << child_stream->get_uid() << ": " << child_stream);
               }

               device_launch_op.stream = child_stream;
            }
         } else {
            assert(size == sizeof(cudaStream_t));
            thread->m_local_mem->read(from_addr, size, &child_stream);

            kernel_info_t & parent_kernel = thread->get_kernel();
            if(child_stream == 0) { //default stream on device for current CTA
               child_stream = parent_kernel.get_default_stream_cta(thread->get_agg_group_id(), thread->get_ctaid()); 
               DEV_RUNTIME_REPORT("launching child kernel " << device_grid->get_uid() << 
                 " to stream " << child_stream->get_uid() << ": " << child_stream);
            }
            else {
               assert(parent_kernel.cta_has_stream(thread->get_agg_group_id(), thread->get_ctaid(), child_stream)); 
               DEV_RUNTIME_REPORT("launching child kernel " << device_grid->get_uid() << 
                 " to stream " << child_stream->get_uid() << ": " << child_stream);
            }
            k_dis->stream = child_stream;
         }
      }

   }

   //launch child kernel
   if(!g_dyn_child_thread_consolidation){
      g_cuda_device_launch_op.push_back(device_launch_op);
   }
   g_cuda_device_launch_param_map.erase(parameter_buffer);

   //set retval0
   const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
   const symbol *formal_return = target_func->get_return_var(); //cudaError_t
   unsigned int return_size = formal_return->get_size_in_bytes();
   DEV_RUNTIME_REPORT("cudaLaunchDeviceV2 return value has size of " << return_size);
   assert(actual_return_op.is_param_local());
   assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size 
     && return_size == sizeof(cudaError_t));
   cudaError_t error = cudaSuccess;
   addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
   thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);

   //Po-Han: DCC implementation: try to merge existing child kernels
   //1) only valid kernels can be merged since they have the parameters
   if(g_dyn_child_thread_consolidation){
      std::list<dcc_kernel_distributor_t>::iterator kd_entry_1, kd_entry_2;
      for( kd_entry_1 = g_cuda_dcc_kernel_distributor.begin(); kd_entry_1 != g_cuda_dcc_kernel_distributor.end(); kd_entry_1++) {
         for( kd_entry_2 = g_cuda_dcc_kernel_distributor.begin(); kd_entry_2 != g_cuda_dcc_kernel_distributor.end(); kd_entry_2++ ) {
            if( kd_entry_1->valid == false || kd_entry_1->launched == true ) {
               break;
            } else if( kd_entry_2->valid == false || kd_entry_2->launched == true ) {
               continue;
            } else if( kd_entry_1 != kd_entry_2 && 
              !kd_entry_1->kernel_grid->name().compare(kd_entry_2->kernel_grid->name()) && 
              kd_entry_1->kernel_grid->get_parent() == kd_entry_2->kernel_grid->get_parent() ){
               //different child kernel, check if they can merge
               bool remained;
               bool merged = merge_two_kernel_distributor_entry( &(*kd_entry_1), &(*kd_entry_2), false, -1, remained );

               if(merged){
                  // invalidate and erase kernel 2
                  kd_entry_2->valid = false;
                  kd_entry_2->kernel_grid->get_parent()->delete_stream_cta(kd_entry_2->agg_group_id, kd_entry_2->ctaid, kd_entry_2->stream); //destroy useless cuda stream
                  g_cuda_dcc_kernel_distributor.erase(kd_entry_2);
//                  kd_entry_2 = g_cuda_dcc_kernel_distributor.erase(kd_entry_2);
                  kd_entry_2 = g_cuda_dcc_kernel_distributor.begin();
                  DEV_RUNTIME_REPORT("DCC: successfully merged, kernel distrubutor now has " << g_cuda_dcc_kernel_distributor.size() << " entries.");
               }
            }
         }
      }
      if(pending_child_threads > per_kernel_optimal_child_size[g_app_name]){
         DEV_RUNTIME_REPORT("DCC: enough child threads (" << per_kernel_optimal_child_size[g_app_name] << "), issue child kernels to reduce param buffer size.");
         launch_one_device_kernel(true, NULL, NULL);
      }
      if(param_buffer_full){
         DEV_RUNTIME_REPORT("DCC: parameter buffer full, issue child kernels to reduce param buffer size.");
         launch_one_device_kernel(true, NULL, NULL);
      }
   }
}


//Handling device runtime api:
//cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int  flags)
//flags can only be cudaStreamNonBlocking
void gpgpusim_cuda_streamCreateWithFlags(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func) {
   DEV_RUNTIME_REPORT("Calling cudaStreamCreateWithFlags");

   unsigned n_return = target_func->has_return();
   assert(n_return);
   unsigned n_args = target_func->num_args();
   assert( n_args == 2 );

   size_t generic_pStream_addr;
   addr_t pStream_addr;
   unsigned int flags;
   for( unsigned arg=0; arg < n_args; arg ++ ) {
      const operand_info &actual_param_op = pI->operand_lookup(n_return+1+arg); //param#
      const symbol *formal_param = target_func->get_arg(arg); //cudaStreamCreateWithFlags_param_#
      unsigned size=formal_param->get_size_in_bytes();
      assert( formal_param->is_param_local() );
      assert( actual_param_op.is_param_local() );
      addr_t from_addr = actual_param_op.get_symbol()->get_address();

      if(arg == 0) {//cudaStream_t * pStream, address of cudaStream_t
         assert(size == sizeof(cudaStream_t *));
         thread->m_local_mem->read(from_addr, size, &generic_pStream_addr);

         //pStream should be non-zero address in local memory
         pStream_addr = generic_to_local(thread->get_hw_sid(), thread->get_hw_tid(), generic_pStream_addr);

         DEV_RUNTIME_REPORT("pStream locating at local memory " << pStream_addr);
      }
      else if(arg == 1) { //unsigned int flags, should be cudaStreamNonBlocking
         assert(size == sizeof(unsigned int));
         thread->m_local_mem->read(from_addr, size, &flags);
         //assert(flags == cudaStreamNonBlocking);
      }
   }

   //create stream and write back to param0
   CUstream_st * stream = thread->get_kernel().create_stream_cta(thread->get_agg_group_id(), thread->get_ctaid());
   DEV_RUNTIME_REPORT("Create stream " << stream->get_uid() << ": " << stream);
   thread->m_local_mem->write(pStream_addr, sizeof(cudaStream_t), &stream, NULL, NULL);

   //set retval0
   const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
   const symbol *formal_return = target_func->get_return_var(); //cudaError_t
   unsigned int return_size = formal_return->get_size_in_bytes();
   DEV_RUNTIME_REPORT("cudaStreamCreateWithFlags return value has size of " << return_size);
   assert(actual_return_op.is_param_local());
   assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size 
     && return_size == sizeof(cudaError_t));
   cudaError_t error = cudaSuccess;
   addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
   thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);

}

void launch_one_device_kernel(bool no_more_kernel, kernel_info_t *fin_parent, ptx_thread_info *sync_parent_thread) {
   if(!g_dyn_child_thread_consolidation){
      if(!g_cuda_device_launch_op.empty()) {
         device_launch_operation_t &op = g_cuda_device_launch_op.front();

         if(op.op_name == DEVICE_LAUNCH_CHILD) {
            stream_operation stream_op = stream_operation(op.grid, g_ptx_sim_mode, op.stream);
            g_stream_manager->push(stream_op);
         }
         else if (op.op_name == DEVICE_LAUNCH_AGG) {
            op.grid->add_agg_block_group(op.agg_block_group); 
         }
         else {
            assert(1 && "Error: device launch operation unrecognized\n");
         }
         g_cuda_device_launch_op.pop_front();
      }
   } else {
      assert (fin_parent == NULL || sync_parent_thread == NULL); //either of these two pointers must be NULL
      dev_launch_type launch_mode = (fin_parent != NULL) ? PARENT_FINISHED : ((sync_parent_thread != NULL) ? PARENT_BLOCK_SYNC : NORMAL);
      bool enough_threads = false;
      bool parent_finished = false;
      bool isPRkernel2 = false;
      size_t found2;
      if ( !g_cuda_dcc_kernel_distributor.empty() ){
         if( launch_mode == NORMAL || launch_mode == PARENT_FINISHED ){
            if( g_cuda_dcc_kernel_distributor.size() > 1 ) {
               g_cuda_dcc_kernel_distributor.sort(compare_dcc_kd_entry);
            }
         }
         std::list<dcc_kernel_distributor_t>::iterator it;

         /* searching for target kernel distributor entry 
          * NORMAL: any entry that is valid 
          * PARENT_FINISHED: any entry that is its child
          * PARENT_BLOCK_SYNC: any entry that is certain block's child */
         bool found_target_entry = false;
         int target_merge_size = -1;
         bool remained = false;
         for(it = g_cuda_dcc_kernel_distributor.begin(); it != g_cuda_dcc_kernel_distributor.end(); it++){ //find a valid kd entry with the most threads
            if(it->valid){ 
               switch(launch_mode){
               case NORMAL:
                  if(g_app_name == BFS || g_app_name == AMR || g_app_name == JOIN || g_app_name == SSSP || g_app_name == BFS_RODINIA || g_app_name == PAGERANK || 
                    (g_app_name == MIS /*&& it->kernel_grid->name().find(mis_k2) != std::string::npos*/) ) {
                     enough_threads = (g_dyn_child_thread_consolidation_version >= 2) ? (pending_child_threads > it->optimal_kernel_size) : (pending_child_threads > it->optimal_block_size);
                     if(no_more_kernel && enough_threads){
                        found_target_entry = true;
                        switch(g_dyn_child_thread_consolidation_version){
                        case 0: // issue a new kernel with exactly 1 block
                           target_merge_size = it->optimal_block_size;
                           break;
                        case 1: // issue a new kernel with as many blocks as possible
                           target_merge_size = pending_child_threads - (pending_child_threads % it->optimal_block_size);
                           break;
                        case 2: // issue a new kernel with optimal kernel size (minimum size that can fill up the whole GPU)
                           target_merge_size = pending_child_threads - (pending_child_threads % it->optimal_block_size);
                           //target_merge_size = it->optimal_kernel_size;
                        default:
                           break;
                        }
                        DEV_RUNTIME_REPORT("DCC: independent child kernel and enough pending child threads => merge for a " << target_merge_size << " threads block.");
                     }
                  }
                  break;
               case PARENT_FINISHED:
                  if (it->kernel_grid->get_parent() == fin_parent){ 
                     found_target_entry = true;
                     assert(it->kernel_grid->get_parent()->end_cycle != 0); //make sure that the parent kernel actually finished
                     target_merge_size = -1;
                     DEV_RUNTIME_REPORT("DCC: parent kernel " << fin_parent->get_uid() << " finished => merge all its child kernels together and issue it");
                  }
                  break;
               case PARENT_BLOCK_SYNC:
                  if (it->kernel_grid->get_parent() == &(sync_parent_thread->get_kernel()) ){
                     std::list<ptx_thread_info *>::iterator parent_it = it->kernel_grid->m_parent_threads.begin();
                     unsigned tmp_block_idx = (*parent_it)->get_block_idx();
                     if( tmp_block_idx == sync_parent_thread->get_block_idx()){                   
                        found_target_entry = true;
                        target_merge_size = -1;
                        DEV_RUNTIME_REPORT("DCC: parent block " << sync_parent_thread->get_block_idx() << " has called cudaDeviceSynchronize => merge all its child kernels and issue it");
                     }
                  }
                  break;
               default:
                  DEV_RUNTIME_REPORT("DCC: unsupported device launch mode");
                  assert(0);
                  break;
               }
               if(found_target_entry) break;
#if 0
               found2 = it->kernel_grid->name().find(pr_k2);
               if(found2 != std::string::npos) isPRkernel2 == true;
               if( g_app_name == COLOR || g_app_name == MIS || (g_app_name == PAGERANK && isPRkernel2) || g_app_name == KMEANS){ //applications with parent-child dependency -> launch it only if its parent block has called cudaDeviceSynchronize
                  kernel_info_t *parent_grid = it->kernel_grid->get_parent();
                  unsigned parent_block_idx = it->kernel_grid->m_parent_threads.front()->get_block_idx();
                  if (parent_grid->block_state[parent_block_idx].switched) break;
                  else continue;
               } else {
                  break;
               }
#endif
            }
         }
         if(!found_target_entry) return; //cannot found kernel distributor entry that fits current launch mode

         if(g_cuda_dcc_kernel_distributor.size() > 1){
            std::list<dcc_kernel_distributor_t>::iterator it2;
            for(it2=g_cuda_dcc_kernel_distributor.begin(); it2!=g_cuda_dcc_kernel_distributor.end(); it2++){
               if( (it2->valid) && (it2!=it) && (!it->kernel_grid->name().compare(it2->kernel_grid->name())) && (it->kernel_grid->get_parent() == it2->kernel_grid->get_parent()) ){ //valid and different
                  if( launch_mode == PARENT_BLOCK_SYNC && (it2->kernel_grid->m_parent_threads.front())->get_block_idx() != sync_parent_thread->get_block_idx() ) { //additional constraint
                     continue;
                  }
                  bool merged = merge_two_kernel_distributor_entry(&(*it), &(*it2), true, target_merge_size, remained);
                  if(merged){
                     if(!remained){
                        // invalidate and erase kernel 2
                        it2->valid = false;
                        it2->kernel_grid->get_parent()->delete_stream_cta(it2->agg_group_id, it2->ctaid, it2->stream); //destroy useless cuda stream
                        it2 = g_cuda_dcc_kernel_distributor.erase(it2);
                        it2--;
                     }
                     DEV_RUNTIME_REPORT("DCC: successfully merged, kernel distrubutor now has " << g_cuda_dcc_kernel_distributor.size() << " entries.");
                  }
                  if ( target_merge_size != -1 && it->thread_count == target_merge_size) break;
               }
            }
         }
         DEV_RUNTIME_REPORT("DCC: launch kernel " << it->kernel_grid->get_uid() << " with " << it->thread_count << " threads, merged from " << it->merge_count << " child kernels, waited "  << gpu_sim_cycle+gpu_tot_sim_cycle - it->kernel_grid->launch_cycle << " cycles, kernel distributor now has " << pending_child_threads << " pending threads.");
         fprintf(stderr, "%llu, %u, %d, %d, %d, %d, %d\n", gpu_tot_sim_cycle+gpu_sim_cycle, it->kernel_grid->get_uid(), it->thread_count, it->merge_count, parent_finished, no_more_kernel, enough_threads);
         it->kernel_grid->reset_block_state();
         pending_child_threads -= it->thread_count;
         stream_operation stream_op = stream_operation(it->kernel_grid, g_ptx_sim_mode, it->stream);
         g_stream_manager->push(stream_op);
         // remove a kernel distributor entry after it is launched
         g_cuda_dcc_kernel_distributor.erase(it);
      } 
   }
}

/* Po-Han: TODO support parent-child synchronization
 * (1) set the calling thread as blocked, wait for the kernel finish function to active it
 */
void gpgpusim_cuda_deviceSynchronize(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func) {
   DEV_RUNTIME_REPORT("Calling cudaDeviceSynchronize");
   unsigned parent_block_idx = thread->get_block_idx();
   std::list<dcc_kernel_distributor_t>::iterator it;
   bool has_child_kernels = false;
   if(g_dyn_child_thread_consolidation){
      for(it=g_cuda_dcc_kernel_distributor.begin(); it!=g_cuda_dcc_kernel_distributor.end(); it++){
         if((it->kernel_grid->m_parent_threads.front())->get_block_idx() == parent_block_idx){
            has_child_kernels = true;
            break;
         }
      }
   }else{
      if(!thread->get_kernel().block_state[parent_block_idx].thread.all()) //a block has child kernel only if some thread turn-off its bit in block state
         has_child_kernels = true;
   }
   /* Parent-child dependency */
   if(/*!thread->get_kernel().block_state[parent_block_idx].devsynced &&*/ has_child_kernels){
      thread->get_kernel().block_state[parent_block_idx].devsynced = true;
      if(!g_dyn_child_thread_consolidation){
         //cdp: context switch current CTA
         DEV_RUNTIME_REPORT("CDP: mark parent kernel " << thread->get_kernel().get_uid() << " block " << parent_block_idx << " for context-switch.");
         if (!thread->get_kernel().block_state[parent_block_idx].switched) {// if this cta is selected for switching first time, set time stamp
            thread->get_kernel().block_state[parent_block_idx].time_stamp_switching = 0;
            thread->get_kernel().block_state[parent_block_idx].switched = 1;
            thread->get_kernel().block_state[parent_block_idx].preempted = 0;
         }
      } else {
         //DCC: register a barrier (borrow from the data structure of context switch) and blocks the whole CTA
         DEV_RUNTIME_REPORT("DCC: mark parent kernel " << thread->get_kernel().get_uid() << " block " << parent_block_idx << " as stalled.");
         if (!thread->get_kernel().block_state[parent_block_idx].switched) {// if this cta is selected for switching first time, set time stamp
            thread->get_kernel().block_state[parent_block_idx].switched = 1;
         }
         launch_one_device_kernel(true, NULL, thread);
      }
      thread->get_kernel().parent_child_dependency = true;
   }else{
      DEV_RUNTIME_REPORT("Useless cudaDeviceSynchronize");
   }

   //copy the buffer address to retval0
   const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
   const symbol *formal_return = target_func->get_return_var(); //void *
   unsigned int return_size = formal_return->get_size_in_bytes();
   DEV_RUNTIME_REPORT("cudaDeviceSynchronize return value has size of " << return_size);
   assert(actual_return_op.is_param_local());
   assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size 
     && return_size == sizeof(cudaError_t));
   cudaError_t error = cudaSuccess;
   addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
   thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);
}

void launch_all_device_kernels() {
   while(!g_cuda_device_launch_op.empty()) {
      launch_one_device_kernel(true, NULL, NULL);
   }
}

kernel_info_t * find_launched_grid(function_info * kernel_entry) {
   if(g_agg_blocks_support) {
      kernel_info_t * grid;
      grid = g_stream_manager->find_grid(kernel_entry);

      if(grid != NULL)
         return grid;
      else {
         for(auto launch_op = g_cuda_device_launch_op.begin(); 
           launch_op != g_cuda_device_launch_op.end();
           launch_op++) {
            if(launch_op->op_name == DEVICE_LAUNCH_CHILD &&
              launch_op->grid->entry() == kernel_entry)
               return launch_op->grid;
         }
         return NULL;
      }
   }
   else {
      return NULL;
   }
}
