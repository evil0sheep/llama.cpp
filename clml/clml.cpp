#include "clml.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <iostream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>  

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            fprintf(stderr, "opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            exit(1);                                                \
        }                                                           \
    } while (0)

#define MULTILINE_QUOTE(...) #__VA_ARGS__

const char* program_source = MULTILINE_QUOTE(

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


__kernel void dequantize_tensor_q4_0(
    const __global uchar* weights, 
    const __global uchar* metadata, 
    __global half* result, 
    const uint2 size, 
    const uint weight_stride, 
    const uint metadata_stride) {

    const uint qk = 32;
    const uint nb = size.x / qk;

    for(uint row_idx = get_group_id(1); row_idx < size.y; row_idx += get_num_groups(1)){
        const __global uchar* weights_row = weights + row_idx * weight_stride;
        const __global uchar* metadata_row = metadata + row_idx * metadata_stride;
        __global half* result_row = result + row_idx * size.x;

        for(uint block_idx = get_local_id(0); block_idx < nb; block_idx += get_local_size(0)){
            const __global uchar* weights_block = weights_row + block_idx * qk/2;
            const __global half16* result_block = (__global half16*) (result_row + block_idx * qk);
            float d = vload_half(0, metadata_row + block_idx * sizeof(half));
            
            const uchar16 qs = vload16(0, weights_block);
            const uchar16 qh = (qs >> (uchar)4) ;
            const uchar16 ql = (qs & (uchar)0xF);

            vstore16((convert_half16(ql)-8) * convert_half(d), 0, result_block);
            vstore16((convert_half16(qh)-8) * convert_half(d), 1, result_block);

        }
    }


};

);

#define WARP_SIZE 16
#define NUM_CORES 4
#define MAX_WARPS_PER_CORE 8


void clml_dequantize_tensor_q4_0(clml_tensor * x, float * y, int k){
    cl_kernel kernel = x->ctx->kernels["dequantize_tensor_q4_0"];

    cl_int err = 0;
    cl_mem y_mem = clCreateBuffer(x->ctx->context, NULL, sizeof(cl_half) * k, NULL, &err); CL_CHECK(err);
    ggml_fp16_t * y_half = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * k);

    cl_uint2 size = {x->ne[0], x->ne[1]};

    printf("size = %u, %u\n", x->ne[0], x->ne[1]);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->weights));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->metadata));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &y_mem));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint2), &size));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &x->weight_stride));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &x->metadata_stride));

    size_t num_workgroups = NUM_CORES * std::min(x->ne[0]/NUM_CORES, (size_t) MAX_WARPS_PER_CORE);
    size_t global_work_size[2] = {WARP_SIZE, num_workgroups};
    size_t local_work_size[2] = {WARP_SIZE, 1};
    CL_CHECK(clEnqueueNDRangeKernel(x->ctx->queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(x->ctx->queue));

    CL_CHECK(clEnqueueReadBuffer(x->ctx->queue, y_mem, CL_TRUE, 0, sizeof(cl_half) * k, y_half, 0, NULL, NULL));
    CL_CHECK(clReleaseMemObject(y_mem));

    for(int i=0; i<k; i++){
        y[i] = ggml_fp16_to_fp32(y_half[i]);
    }
    free(y_half);
}

void clml_init(clml_context *ctx_out){
    cl_int err = 0;
    cl_platform_id platform;
    CL_CHECK(clGetPlatformIDs(1, &platform, NULL));

    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &ctx_out->device, NULL));

    ctx_out->context = clCreateContext(NULL, 1, &ctx_out->device, NULL, NULL, &err); CL_CHECK(err);
    ctx_out->queue = clCreateCommandQueue(ctx_out->context , ctx_out->device, 0, &err); CL_CHECK(err);

    size_t size;
    clGetDeviceInfo(ctx_out->device, CL_DEVICE_EXTENSIONS, 0, NULL, &size);
    char *extensions = (char*)malloc(size);
    clGetDeviceInfo(ctx_out->device, CL_DEVICE_EXTENSIONS, size, extensions, NULL);

    if (strstr(extensions, "cl_khr_fp16") != NULL) {
       printf("cl_khr_fp16 extension available\n");
    }
    free(extensions);

    size_t log_size;
    char *build_log;

    cl_program program = clCreateProgramWithSource(ctx_out->context, 1, &program_source, NULL, &err); CL_CHECK(err);
    err = clBuildProgram(program, 1, &ctx_out->device, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("clBuildProgram failed\n");
        CL_CHECK(clGetProgramBuildInfo(program, ctx_out->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));

        build_log = (char *)malloc(log_size + 1);
        CL_CHECK(clGetProgramBuildInfo(program, ctx_out->device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
        build_log[log_size] = '\0';
        printf("Build Log:\n%s\n", build_log);
        free(build_log);

        CL_CHECK(err);
    }

   

    ctx_out->kernels["dequantize_tensor_q4_0"] = clCreateKernel(program, "dequantize_tensor_q4_0", &err); CL_CHECK(err);
}

clml_tensor clml_tensor_from_ggml(clml_context *ctx, const ggml_tensor *in){

    printf("name = %s, type = %s, dims =[", in->name, ggml_type_name(in->type));
    for (int i = 0; i < in->n_dims; ++i) {
        printf("%s%ld", i == 0 ? "" : ", ", in->ne[i]);
    }
    printf("]\n");
    
    
    clml_tensor out;
    out.ctx = ctx;
    out.type = in->type;
    out.n_dims = in->n_dims;
    assert(in->n_dims <= 2);
    for(int i=0; i<in->n_dims; i++){
        out.ne[i] = in->ne[i];
    }
    for(int i=in->n_dims; i<GGML_MAX_DIMS; i++){
        out.ne[i] = 1;
    }
    strcpy(out.name, in->name);

    size_t weights_size, metadata_size;
    size_t cache_line_size = 64;
    switch (in->type) {
        case GGML_TYPE_Q4_0:
            out.weight_stride = (QK4_0 / 2) * (in->ne[0] / QK4_0);
            out.metadata_stride = (sizeof(ggml_fp16_t) * in->ne[0]) / QK4_0;
            break;

        case GGML_TYPE_F32:
            out.weight_stride = sizeof(ggml_fp16_t)* in->ne[0];
            out.metadata_stride = 0;
            break;

        case GGML_TYPE_Q6_K:
            out.weight_stride = (QK_K / 2 + QK_K / 4) * (in->ne[0] / QK_K);
            out.metadata_stride = (QK_K/16 + sizeof(ggml_fp16_t)) * (in->ne[0] / QK_K);
            break;
        default:
            fprintf(stderr, "%s: unsupported type %s\n", __func__, ggml_type_name(in->type));
            exit(1);
    }
    if(out.weight_stride % cache_line_size != 0){
        out.weight_stride += cache_line_size - (out.weight_stride % cache_line_size);
    }
    if(out.metadata_stride > 0 && out.metadata_stride % cache_line_size != 0){
        out.metadata_stride += cache_line_size - (out.metadata_stride % cache_line_size);
    }
    assert(out.weight_stride % cache_line_size == 0);
    assert(out.metadata_stride % cache_line_size == 0);

    weights_size = out.weight_stride;
    metadata_size = out.metadata_stride;
    for(int i=1; i<in->n_dims; i++){
        weights_size *= in->ne[i];
        metadata_size *= in->ne[i];
    }

    // printf("weight_stride = %zu, metadata_stride = %zu\n", out.weight_stride, out.metadata_stride);
    // printf("weights_size = %zu, metadata_size = %zu\n", weights_size, metadata_size);

    cl_int err = 0;
    out.weights = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY , weights_size, NULL, &err); CL_CHECK(err);
    void *weights_host = malloc(weights_size);
    void *metadata_host = NULL;

    if(metadata_size > 0){
        out.metadata = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, metadata_size, NULL, &err); CL_CHECK(err);
        metadata_host = malloc(metadata_size);
        
    }else{
        out.metadata = NULL;
    }

    assert(in->n_dims <= 2);
    size_t num_rows = in->n_dims == 1 ? 1 : in->ne[1];
    switch (in->type) {
        case GGML_TYPE_Q4_0:
            {
                for(size_t i = 0; i < num_rows; i++){
                    block_q4_0 * in_row = (block_q4_0 *) (in->data + i * in->nb[1]);
                    uint8_t * out_weights_row = ((uint8_t *) weights_host) + i * out.weight_stride;
                    uint8_t * out_metadata_row = ((uint8_t *) metadata_host) + i * out.metadata_stride;
                    for(size_t j = 0; j < in->ne[0]/QK4_0; j++){
                        block_q4_0 * in_block = &in_row[j];
                        memcpy(out_weights_row + j * (QK4_0/2), in_block->qs, QK4_0 / 2);
                        memcpy(out_metadata_row + j * (sizeof(ggml_fp16_t)), &in_block->d, sizeof(ggml_fp16_t));
                    }
                }
            }
            break;

        case GGML_TYPE_F32:
            break;

        case GGML_TYPE_Q6_K:
            break;
        default:
            fprintf(stderr, "%s: unsupported type %s\n", __func__, ggml_type_name(in->type));
            exit(1);
    }

    CL_CHECK(clEnqueueWriteBuffer(ctx->queue, out.weights, CL_TRUE, 0, weights_size, weights_host, 0, NULL, NULL));
    free(weights_host);
    if(metadata_size > 0){
        CL_CHECK(clEnqueueWriteBuffer(ctx->queue, out.metadata, CL_TRUE, 0, metadata_size, metadata_host, 0, NULL, NULL));
        free(metadata_host);
    }
    
    return out;

} 

void clml_free_tensor(clml_tensor * x){
    CL_CHECK(clReleaseMemObject(x->weights));
    if(x->metadata != NULL){
        CL_CHECK(clReleaseMemObject(x->metadata));
    }
}