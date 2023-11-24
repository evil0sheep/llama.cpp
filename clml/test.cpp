#include "clml.h"
#include "ggml.h"
#include "stdio.h"

#include <math.h>

#undef NDEBUG
#include <cassert>  

static void quantize_row_q4_0_reference(const float * x, block_q4_0 * y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = ggml_fp32_to_fp16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

static void dequantize_row_q4_0(const block_q4_0 * x, float * y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void test_interleave(clml_context *clctx){
    printf("\n%s\n", __func__ );

    constexpr size_t n = 1024 * 1024;
    float *input = (float *) malloc(n * sizeof(float));
    float *output_cpu = (float *) malloc(n * sizeof(float));
    float *output_gpu = (float *) malloc(n * sizeof(float));
    for(int i=0; i<n; i++){
        input[i] = i/100.0f;
    }

    struct ggml_init_params params = {
        .mem_size   = 4 * n * sizeof(float),
        .mem_buffer = NULL,
    };
    struct ggml_context * ggctx = ggml_init(params);
    ggml_tensor * ggt = ggml_new_tensor_1d(ggctx, GGML_TYPE_Q4_0, n);

    // block_q4_0 q4_row[n/QK4_0];
    
    quantize_row_q4_0_reference(input, (block_q4_0 *)ggt->data, n);
    dequantize_row_q4_0((block_q4_0 *)ggt->data, output_cpu, n);

    // for(int i=0; i<n; i++){
    //   printf("%f %f\n", input[i], output_cpu[i]);
    // }

    clml_tensor clt = clml_tensor_from_ggml(clctx, ggt);

    clml_dequantize_tensor_q4_0(&clt, output_gpu, n);

    float max_diff_percent = 0;
    size_t max_diff_idx = 0;
    for(int i=0; i<n; i++){
        float diff_percent =  fabsf((output_cpu[i] - output_gpu[i])/ output_cpu[i]);
        if(diff_percent > max_diff_percent){
        max_diff_percent = diff_percent;
        max_diff_idx = i;
        }
    }
    printf("max_diff = %f at %zu (%f vs %f)\n", max_diff_percent, max_diff_idx, output_cpu[max_diff_idx], output_gpu[max_diff_idx]);

    free(input);
    free(output_cpu);
    free(output_gpu);
    ggml_free(ggctx);
    clml_free_tensor(&clt);
}


void test_interleave_2d(clml_context *clctx){
    printf("\n%s\n", __func__ );
    constexpr size_t n = 1024;
    float *input = (float *) malloc(n * n * sizeof(float));
    float *output_cpu = (float *) malloc(n * n * sizeof(float));
    float *output_gpu = (float *) malloc(n * n * sizeof(float));
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            size_t idx = i*n + j;
            input[idx] = idx/100.0f;        }
    }

    struct ggml_init_params params = {
        .mem_size   = 4 * n * n * sizeof(float),
        .mem_buffer = NULL,
    };
    struct ggml_context * ggctx = ggml_init(params);
    ggml_tensor * ggt = ggml_new_tensor_2d(ggctx, GGML_TYPE_Q4_0, n, n);

    // block_q4_0 q4_row[n/QK4_0];
    
    for(int i=0; i<n; i++){
        quantize_row_q4_0_reference(input + i*n, (block_q4_0 *)ggt->data + i*n/QK4_0, n);
        dequantize_row_q4_0((block_q4_0 *)ggt->data + i*n/QK4_0, output_cpu + i*n, n);
    }

    // for(int i=0; i<n * n; i++){
    //   printf("%f %f\n", input[i], output_cpu[i]);
    // }

    clml_tensor clt = clml_tensor_from_ggml(clctx, ggt);

    clml_dequantize_tensor_q4_0(&clt, output_gpu, n * n);

    float max_diff_percent = 0;
    size_t max_diff_idx = 0;
    for(int i=0; i<n * n; i++){
        float diff_percent =  fabsf((output_cpu[i] - output_gpu[i])/ output_cpu[i]);
        if(diff_percent > max_diff_percent){
        max_diff_percent = diff_percent;
        max_diff_idx = i;
        }
    }
    printf("max_diff = %f at %zu (%f vs %f)\n", max_diff_percent, max_diff_idx, output_cpu[max_diff_idx], output_gpu[max_diff_idx]);

    free(input);
    free(output_cpu);
    free(output_gpu);
    ggml_free(ggctx);
    clml_free_tensor(&clt);
}

int main(void){
  clml_context ctx;
  clml_init(&ctx);

  test_interleave(&ctx);
  test_interleave_2d(&ctx);
}