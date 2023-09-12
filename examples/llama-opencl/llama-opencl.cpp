
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "clml/clml.h"

#include <iostream>
#include <string>
#include <vector>


// read and create ggml_context containing the tensors and their data
bool gguf_load_model(const std::string & fname, clml_context *clml_context) {
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    printf("%s: version:      %d\n", __func__, gguf_get_version(ctx));
    printf("%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    printf("%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        printf("%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);
            const enum gguf_type kv_type = gguf_get_kv_type(ctx, i);

            printf("%s: kv[%d]: key = %s, type = %s\n", __func__, i, key, gguf_type_name(kv_type));
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        printf("%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            printf("%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }

    // data
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            // printf("%s: reading tensor %d data\n", __func__, i);

            const char * name = gguf_get_tensor_name(ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);


            struct clml_tensor clml_tensor = clml_tensor_from_ggml(clml_context, cur);
            // print first 10 elements
            // const float * data = (const float *) cur->data;

            // printf("%s data[:10] : ", name);
            // for (int j = 0; j < MIN(10, ggml_nelements(cur)); ++j) {
            //     printf("%f ", data[j]);
            // }
            // printf("\n\n");

            // // check data
            // {
            //     const float * data = (const float *) cur->data;
            //     for (int j = 0; j < ggml_nelements(cur); ++j) {
            //         if (data[j] != 100 + i) {
            //             fprintf(stderr, "%s: tensor[%d]: data[%d] = %f\n", __func__, i, j, data[j]);
            //             return false;
            //         }
            //     }
            // }
        }
    }

    printf("%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(ctx_data));

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        printf("usage: %s data.gguf\n", argv[0]);
        return -1;
    }

    const std::string fname(argv[1]);

    std::cout << "loading file " << fname << std::endl;

    clml_context clml_context;
    clml_init(&clml_context);

    gguf_load_model(fname, &clml_context);

}
