#include <CL/cl.h>
#include "ggml.h"

#include <unordered_map>
#include <string>

#define QK4_0 32
#define QK_K 256

typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_fp16_t d;           // super-block scale
} block_q6_K;

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    std::unordered_map<std::string, cl_kernel> kernels;
} clml_context;


typedef struct {
        enum ggml_type    type;

        int     n_dims;
        size_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  weight_stride;                      
        size_t  metadata_stride;

        cl_mem weights;
        cl_mem metadata;

        clml_context *ctx;

        char name[GGML_MAX_NAME];
} clml_tensor;

void clml_dequantize_tensor_q4_0(clml_tensor * x, float * y, int k);

void clml_init(clml_context * ctx_out);

clml_tensor clml_tensor_from_ggml(clml_context *ctx, const ggml_tensor *in );