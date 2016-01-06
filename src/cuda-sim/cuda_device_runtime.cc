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
  MST,
  MIS,
  PAGERANK,
  KMEANS
  } application_id;*/

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
application_id g_app_name = BFS;

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
      g_total_param_size += child_kernel_arg_size; 
      DEV_RUNTIME_REPORT("DCC: child kernel arg pre-allocation: size " << child_kernel_arg_size << ", parameter buffer allocated at " << param_buffer);
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

      // initialize the kernel distributor entry
      dcc_kernel_distributor_t distributor_entry(device_grid, total_thread_count, optimal_threads_per_block, optimal_threads_per_kernel, param_buffer);
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
   mspace1 = kd_entry_1->kernel_grid->get_param_memory(-1);
   mspace2 = kd_entry_2->kernel_grid->get_param_memory(-1);
   unsigned int total_thread_1, total_thread_2;
   unsigned int offset_a_1, offset_a_2, offset_b_1, offset_b_2;
   unsigned int total_thread_sum, total_thread_offset;
   unsigned int kernel_param_size;
   unsigned int new_offset_b_1, new_offset_b_2;
   dim3 gDim;

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
      mspace1->read((size_t)36, 4, &offset_a_1);
      mspace2->read((size_t)36, 4, &offset_a_2);

      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

      if( offset_a_1 + total_thread_1 == offset_a_2 ){
         DEV_RUNTIME_REPORT("DCC: AMR continous -> child1 (" << offset_a_1 << ", " << total_thread_1 << ") child2 (" << offset_a_2 << ", " << total_thread_2 << ")");
         continous_offset = true;
      }
      total_thread_offset = 0;
      kernel_param_size = 52;
      break;
   case JOIN:
      //[base_a (8B), base_b (8B), offset_a (4B), total_thread (4B), offset_b (4B), var (4B)]
      mspace1->read((size_t)16, 4, &offset_a_1);
      mspace2->read((size_t)16, 4, &offset_a_2);
      mspace1->read((size_t)20, 4, &total_thread_1);
      mspace2->read((size_t)20, 4, &total_thread_2);
      mspace1->read((size_t)24, 4, &offset_b_1);
      mspace2->read((size_t)24, 4, &offset_b_2);
      
      assert( kd_entry_1->thread_count == total_thread_1 );
      assert( kd_entry_2->thread_count == total_thread_2 );

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
      kernel_param_size = 56;
      break;
   case MST:
      break;
   case MIS:
      break;
   case PAGERANK:
      break;
   case KMEANS:
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
   default:
      DEV_RUNTIME_REPORT("Dynamic Child-thread Consolidation: unsupported application");
      assert(0);
      break;
   }

   unsigned remaining_count = 0;
   std::map<unsigned int, memory_space *>::iterator it;

   if(continous_offset || ForceMerge){
      //child kernel 2 can be catenated after child kernel 1

      // adjust thread count
      total_thread_sum = total_thread_1 + total_thread_2;
      if( (target_size != -1) && (total_thread_sum > target_size) ){
         if (kd_entry_2->kernel_grid->m_param_mem_map.size() > 1) //if the latter kd entry is composed of more than 1 child kernel, find another one
            return false;
         remaining = true;
         remaining_count = total_thread_sum - target_size; //record the number of threads that should be cut-off
         new_mspace = new memory_space_impl<256>("param", 256);
         //copy the kernel parameters of the splitting kernel into a new memory space
         it = kd_entry_2->kernel_grid->m_param_mem_map.begin();
         for(unsigned n = 0; n < kernel_param_size; n += 4) {
            unsigned int oneword;
            it->second->read((size_t) n, 4, &oneword);
            new_mspace->write((size_t) n, 4, &oneword, NULL, NULL); 
         }
      }
      total_thread_sum -= remaining_count;
      kd_entry_1->thread_count = total_thread_sum;
      mspace1->write((size_t)total_thread_offset, 4, &total_thread_sum, NULL, NULL);
      mspace2->write((size_t)total_thread_offset, 4, &total_thread_sum, NULL, NULL);
      if(remaining){
         new_mspace->write((size_t)total_thread_offset, 4, &remaining_count, NULL, NULL);
         kd_entry_2->thread_count = remaining_count;
         gDim = kd_entry_2->kernel_grid->get_grid_dim(-1);
         gDim.x = 1;
         kd_entry_2->kernel_grid->set_grid_dim(gDim);
         DEV_RUNTIME_REPORT("Merge block with " << total_thread_2 << " threads into block with " << total_thread_1 << " under target block size " << total_thread_sum << ", split.");
      }
      

      // adjust grid dimension
      gDim = kd_entry_1->kernel_grid->get_grid_dim(-1);
      unsigned int thread_per_block = kd_entry_1->kernel_grid->threads_per_cta();
      unsigned int num_blocks = (total_thread_sum + thread_per_block - 1) / thread_per_block;
      gDim.x = num_blocks;
      kd_entry_1->kernel_grid->set_grid_dim(gDim);

      // set up launch cycle
      kd_entry_1->kernel_grid->launch_cycle = gs_min2(kd_entry_1->kernel_grid->launch_cycle, kd_entry_2->kernel_grid->launch_cycle);
//      DEV_RUNTIME_REPORT("DCC: merge child kernel " << kd_entry_2->kernel_grid->get_uid() << " into child kernel " << kd_entry_1->kernel_grid->get_uid() << ", new threads " << total_thread_sum << ", new blocks " << num_blocks);

      // merge parameter buffer
      for( it = kd_entry_2->kernel_grid->m_param_mem_map.begin(); it != kd_entry_2->kernel_grid->m_param_mem_map.end(); it++ ){
         unsigned int offset = it->first + total_thread_1;
         kd_entry_1->kernel_grid->m_param_mem_map[offset] = it->second;
         it->second->write((size_t)total_thread_offset, 4, &total_thread_sum, NULL, NULL);
         switch(g_app_name){
         case BFS:
            it->second->read((size_t) 0, 4, &offset_b_1);
            it->second->read((size_t)24, 4, &offset_b_2);
            if(remaining){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_offset_b_2 = offset_b_2 + total_thread_2 - remaining_count;
               new_mspace->write((size_t) 0, 4, &new_offset_b_1, NULL, NULL);
               new_mspace->write((size_t)24, 4, &new_offset_b_2, NULL, NULL);
            }
            offset_b_1 -= total_thread_1;
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            offset_b_2 -= total_thread_1;
            it->second->write((size_t)24, 4, &offset_b_2, NULL, NULL);
            break;
         case AMR:
            it->second->read((size_t)36, 4, &offset_b_1);
            if(remaining){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_mspace->write((size_t)36, 4, &new_offset_b_1, NULL, NULL);
            }
            offset_b_1 -= total_thread_1;
            it->second->write((size_t)36, 4, &offset_b_1, NULL, NULL);
            break;
         case JOIN:
            it->second->read((size_t)16, 4, &offset_b_1);
            it->second->read((size_t)24, 4, &offset_b_2);
            if(remaining){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_offset_b_2 = offset_b_2 + total_thread_2 - remaining_count;
               new_mspace->write((size_t)16, 4, &new_offset_b_1, NULL, NULL);
               new_mspace->write((size_t)24, 4, &new_offset_b_2, NULL, NULL);
            }
            offset_b_1 -= total_thread_1;
            it->second->write((size_t)16, 4, &offset_b_1, NULL, NULL);
            offset_b_2 -= total_thread_1;
            it->second->write((size_t)24, 4, &offset_b_2, NULL, NULL);
            break;
         case SSSP:
            it->second->read((size_t) 0, 4, &offset_b_1);
            if(remaining){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_mspace->write((size_t) 0, 4, &new_offset_b_1, NULL, NULL);
            }
            offset_b_1 -= total_thread_1;
            it->second->write((size_t) 0, 4, &offset_b_1, NULL, NULL);
            break;
         case MST:
            break;
         case MIS:
            break;
         case PAGERANK:
            break;
         case KMEANS:
            break;
         case BFS_RODINIA:
            it->second->read((size_t) 8, 4, &offset_b_1);
            if(remaining){
               new_offset_b_1 = offset_b_1 + total_thread_2 - remaining_count;
               new_mspace->write((size_t) 8, 4, &new_offset_b_1, NULL, NULL);
            }
            offset_b_1 -= total_thread_1;
            it->second->write((size_t) 8, 4, &offset_b_1, NULL, NULL);
            break;
         default:
            DEV_RUNTIME_REPORT("Dynamic Child-thread Consolidation: unsupported application");
            assert(0);
            break;
         }
         DEV_RUNTIME_REPORT("DCC: copy kernel param " << it->second << " old offset " << it->first << " new offset " << offset );
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
         kd_entry_1->kernel_grid->get_parent()->remove_child(kd_entry_2->kernel_grid);
      } else {
         std::list<ptx_thread_info *>::iterator mpt_it;
         for(mpt_it = kd_entry_2->kernel_grid->m_parent_threads.begin(); mpt_it != kd_entry_2->kernel_grid->m_parent_threads.end(); mpt_it++){
            kd_entry_1->kernel_grid->add_parent(kd_entry_2->kernel_grid->get_parent(), *mpt_it);
         }
         // reset the parameter map of the second kd entry and linked it with new memory space
         kd_entry_2->kernel_grid->m_param_mem_map.clear();
         kd_entry_2->kernel_grid->m_param_mem_map[remaining_count] = new_mspace;
         kd_entry_2->kernel_grid->set_param_mem(new_mspace);
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
                  DEV_RUNTIME_REPORT("DCC: activate kernel distributor entry " << i << " with parameter buffer address " << parameter_buffer);
                  break;
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
            } else if( kd_entry_1 != kd_entry_2 ){
               //different child kernel, check if they can merge
               bool remained;
               bool merged = merge_two_kernel_distributor_entry( &(*kd_entry_1), &(*kd_entry_2), false, -1, remained );

               if(merged){
                  // invalidate and erase kernel 2
                  kd_entry_2->valid = false;
                  kd_entry_2 = g_cuda_dcc_kernel_distributor.erase(kd_entry_2);
                  kd_entry_2--;
                  DEV_RUNTIME_REPORT("DCC: successfully merged, kernel distrubutor now has " << g_cuda_dcc_kernel_distributor.size() << " entries.");
               }
            }
         }
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

void launch_one_device_kernel(bool no_more_kernel) {
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
      bool enough_threads = false;
      bool parent_finished = false;
      if ( !g_cuda_dcc_kernel_distributor.empty() ){
         if( g_cuda_dcc_kernel_distributor.size() > 1 ) {
            g_cuda_dcc_kernel_distributor.sort(compare_dcc_kd_entry);
         }
         std::list<dcc_kernel_distributor_t>::iterator it = g_cuda_dcc_kernel_distributor.begin();
         enough_threads = (g_dyn_child_thread_consolidation_version == 2) ? (pending_child_threads > it->optimal_kernel_size) : (pending_child_threads > it->optimal_block_size);
         parent_finished = (it->kernel_grid->get_parent()->end_cycle != 0);

         for(it = g_cuda_dcc_kernel_distributor.begin(); it != g_cuda_dcc_kernel_distributor.end(); it++) //find a valid kd entry with the most threads
            if(it->valid) break;
         if(it == g_cuda_dcc_kernel_distributor.end()) return; //all kd entries are invalid

         /* Po-Han DCC
          * Parent kernel finished 
          *    --> merge all child threads together
          * Parent kernel not finished
          *    --> enough pending thread && no more blocks can be issued from existing kernels
          *       --> 1) merge until pending threads < optimal block size
          *       --> 2> merge and output a kernel with optimal block size
          *    --> otherwise, do nothing
          * */
         bool issuing_kernel = false;
         bool forcing_merge = false;
         int target_merge_size = -1;
         bool remained;
         if(parent_finished){
            issuing_kernel = true;
            forcing_merge = true;
            DEV_RUNTIME_REPORT("DCC: parent kernel finished => merge all child kernels together and issue it");
         }else if(no_more_kernel && enough_threads){
            issuing_kernel = true;
            forcing_merge = true;
            switch(g_dyn_child_thread_consolidation_version){
            case 0: // issue a new kernel with exactly 1 block
               target_merge_size = it->optimal_block_size;
               break;
            case 1: // issue a new kernel with as many blocks as possible
               target_merge_size = pending_child_threads - (pending_child_threads % it->optimal_block_size);
               break;
            case 2: // issue a new kernel with optimal kernel size (exact size that can fill up the whole GPU)
               target_merge_size = it->optimal_kernel_size;
            default:
               break;
            }
            DEV_RUNTIME_REPORT("DCC: parent kernel running but enough pending child threads => merge for a " << target_merge_size << " threads block.");
         }

         if(issuing_kernel){
            if(g_cuda_dcc_kernel_distributor.size() > 1){
               std::list<dcc_kernel_distributor_t>::iterator it2;
               for(it2=g_cuda_dcc_kernel_distributor.begin(); it2!=g_cuda_dcc_kernel_distributor.end(); it2++){
                  if(it2->valid && it2!=it){ //valid and different --> merge
                     bool merged = merge_two_kernel_distributor_entry(&(*it), &(*it2), forcing_merge, target_merge_size, remained);
                     if(merged){
                        if(!remained){
                           // invalidate and erase kernel 2
                           it2->valid = false;
                           it2 = g_cuda_dcc_kernel_distributor.erase(it2);
                           it2--;
                        }
                        DEV_RUNTIME_REPORT("DCC: successfully merged, kernel distrubutor now has " << g_cuda_dcc_kernel_distributor.size() << " entries.");
                     }
                     if (it->thread_count == target_merge_size) break;
                  }
               }
            }
            DEV_RUNTIME_REPORT("DCC: launch kernel " << it->kernel_grid->get_uid() << " with " << it->thread_count << " threads, merged from " << it->merge_count << " child kernels, waited "  << gpu_sim_cycle+gpu_tot_sim_cycle - it->kernel_grid->launch_cycle << " cycles.");
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
}

/* Po-Han: TODO support parent-child synchronization
 * (1) set the calling thread as blocked, wait for the kernel finish function to active it
 */
void gpgpusim_cuda_deviceSynchronize(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func) {
   DEV_RUNTIME_REPORT("Calling cudaDeviceSynchronize");
   unsigned parent_block_idx = thread->get_block_idx();
   if(!g_dyn_child_thread_consolidation){
      //cdp: context switch current CTA
      if (thread->get_kernel().block_state[parent_block_idx].switched) {// if this cta is selected for switching first time, set time stamp
         thread->get_kernel().block_state[parent_block_idx].time_stamp_switching = 0;
         thread->get_kernel().block_state[parent_block_idx].switched = 1;
         thread->get_kernel().block_state[parent_block_idx].preempted = 0;
      }
   } else {
      //DCC: register a barrier (borrow from the data structure of context switch) and blocks the whole CTA
      if (thread->get_kernel().block_state[parent_block_idx].switched) {// if this cta is selected for switching first time, set time stamp
         thread->get_kernel().block_state[parent_block_idx].switched = 1;
      }
   }
   thread->get_kernel().parent_child_dependency = true;
}

void launch_all_device_kernels() {
   while(!g_cuda_device_launch_op.empty()) {
      launch_one_device_kernel(true);
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
