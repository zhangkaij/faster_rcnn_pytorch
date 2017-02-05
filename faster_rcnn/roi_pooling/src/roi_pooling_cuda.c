#include <THC/THC.h>
#include <math.h>
#include "cuda/roi_pooling_kernel.h"

extern THCState *state;

int roi_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, THCudaIntTensor * argmax)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);
    int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, features, 0);
    if (batch_size != 1)
    {
        return 0;
    }
    // data height
    int data_height = THCudaTensor_size(state, features, 2);
    // data width
    int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    ROIPoolForwardLaucher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat,
        output_flat, argmax_flat);

    return 1;
}