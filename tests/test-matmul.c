#include <stdio.h>
#include <string.h>

#include "k_quants.h"
#include "ggml-opencl.h"

# define N 256
int main(void){
    ggml_cl_init();

  float x[N];
  float y[N];
  float s;

  float a[N * N];
  float b[N];
  float c[N];

  memset(x, 0, N * sizeof(float));
  memset(y, 0, N * sizeof(float));

  memset(a, 0, N * N * sizeof(float));
  memset(b, 0, N * sizeof(float));
  memset(c, 0, N * sizeof(float));

  for(int i = 0; i < N; i++){
    x[i] = i * 0.001f;
    y[i] = i;

    for(int j = 0; j < N; j++){
      a[i * N + j] = i  * j;
    }
    b[i] = 1;
    c[i] = -4;
  }

  block_q4_K xq4;
  block_q6_K xq6;
  quantize_row_q4_K(x, &xq4, N);
  quantize_row_q6_K(x, &xq6, N);
  
  block_q8_K yq8;
  quantize_row_q8_K(y, &yq8, N);

  block_q4_K aq4[N];
  for(int i = 0; i < N; i++){

    quantize_row_q4_K(&a[N * i], &aq4[i], N);
  }
  block_q8_K bq8;
  quantize_row_q8_K(b, &bq8, N);
  
  
  s = -42;
  ggml_vec_dot_q4_K_q8_K(N, &s, &xq4, &yq8);

  printf("x dot y cpu returned %f\n", s);

  printf("AxB cpu returned: (");
  for(int i = 0; i < N; i++){
    
    ggml_vec_dot_q4_K_q8_K(N, &c[i], &aq4[i], &bq8);
    printf("%f,\n", c[i]);
  } 
  printf(")\n");

  struct ggml_init_params iparams = {
          .mem_size   = 1024*1024*1024,
          .mem_buffer = NULL,
      };
  struct ggml_context * ctx = ggml_init(iparams);

  struct ggml_tensor * xt4 = ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_K, N);  
  struct ggml_tensor * xt6 = ggml_new_tensor_1d(ctx, GGML_TYPE_Q6_K, N);
  struct ggml_tensor * yt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
  struct ggml_tensor * st = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

  struct ggml_tensor * at = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, N, N);
  struct ggml_tensor * bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
  struct ggml_tensor * ct = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);

  memcpy(xt4->data, &xq4, sizeof(xq4));
  memcpy(xt6->data, &xq6, sizeof(xq6));
  memcpy(yt->data, &y, sizeof(y));



  for(int i = 0; i<N/2; i++){
    printf("%u, ", xq4.qs[i]);
  }
  printf("\n");

  for(int i = 0; i<12; i++){
    printf("%u, ", xq4.scales[i]);
  }
  printf("\n");

  printf("q4: d: %f, dmin: %f\n", xq4.d, xq4.dmin);
  printf("q6: d: %f\n", xq6.d);


  
  ggml_cl_mul_mat(xt4, yt, st, NULL,0);


  printf("ggml_cl_mul_mat returned %f\n", ((float*)st->data)[0]);


   memcpy(at->data, &aq4, sizeof(aq4));
  memcpy(bt->data, &b, sizeof(b));


  ggml_cl_mul_mat(at, bt, ct, NULL,0);

  float * ptr = (float *)ct->data;
  for(int i = 0; i<N; i++){
    printf("%f, ", ptr[i]);
  }
  printf("\n");
}