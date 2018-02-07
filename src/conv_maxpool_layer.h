#ifndef CONV_MAXPOOL_LAYER_H
#define CONV_MAXPOOL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
typedef layer convolutional_layer;
typedef layer maxpool_layer;
typedef layer conv_maxpool_layer;

#ifdef __cplusplus
extern "C" {
#endif

void forward_conv_maxpool_layer(convolutional_layer conv_l, maxpool_layer maxpool_l, network_state state);

#ifdef __cplusplus
}
#endif

#endif /* CONV_MAXPOOL_LAYER_H */
