#include "conv_maxpool_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>



inline void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void forward_conv_maxpool_layer(convolutional_layer conv_l, maxpool_layer maxpool_l, network_state state)
{
    int out_h = conv_l.out_h;
    int out_w = conv_l.out_w;
    int i;

    fill_cpu(conv_l.outputs*conv_l.batch, 0, conv_l.output, 1);

    #if 0
    if(conv_l.xnor){
        binarize_weights(conv_l.weights, conv_l.n, conv_l.c*conv_l.size*conv_l.size, conv_l.binary_weights);
        swap_binary(&conv_l);
        binarize_cpu(state.input, conv_l.c*conv_l.h*conv_l.w*conv_l.batch, conv_l.binary_input);
        state.input = conv_l.binary_input;
    }
    #endif

    int m = conv_l.n;
    int k = conv_l.size*conv_l.size*conv_l.c;
    int n = out_h*out_w;


    float *a = conv_l.weights;
    float *b = state.workspace;
    float *c = conv_l.output;

    for(i = 0; i < conv_l.batch; ++i){
        im2col_cpu(state.input, conv_l.c, conv_l.h, conv_l.w, 
                conv_l.size, conv_l.stride, conv_l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        state.input += conv_l.c*conv_l.h*conv_l.w;
    }

    if(conv_l.batch_normalize){
        forward_batchnorm_layer(conv_l, state);
    } else {
        add_bias(conv_l.output, conv_l.biases, conv_l.batch, conv_l.n, out_h*out_w);
    }

    activate_array(conv_l.output, m*n*conv_l.batch, conv_l.activation);
    
    #if 0
    if(conv_l.binary || conv_l.xnor) swap_binary(&conv_l);
    #endif
    
    {
    state.input = conv_l.output;
    state.index++;
    
    int b,i,j,k,m,n;
    int w_offset = -maxpool_l.pad;
    int h_offset = -maxpool_l.pad;

    int h = maxpool_l.out_h;
    int w = maxpool_l.out_w;
    int c = maxpool_l.c;

    for(b = 0; b < maxpool_l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < maxpool_l.size; ++n){
                        for(m = 0; m < maxpool_l.size; ++m){
                            int cur_h = h_offset + i*maxpool_l.stride + n;
                            int cur_w = w_offset + j*maxpool_l.stride + m;
                            int index = cur_w + maxpool_l.w*(cur_h + maxpool_l.h*(k + b*maxpool_l.c));
                            int valid = (cur_h >= 0 && cur_h < maxpool_l.h &&
                                         cur_w >= 0 && cur_w < maxpool_l.w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    maxpool_l.output[out_index] = max;
                    maxpool_l.indexes[out_index] = max_i;
                }
            }
        }
    }        
    }
}

