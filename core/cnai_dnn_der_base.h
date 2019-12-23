#ifndef CNAI_DNN_DER_BASE_H
#define CNAI_DNN_DER_BASE_H

#include "cnai_inc.h"


REGISTER_OP("SoftplusDeri")
    .Input("x: float")
    .Output("dx: float");

class SoftplusDeri : public OpKernel
{
public:
    explicit SoftplusDeri(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override
    {
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->flat<float>();

        /* NOT Implemented */
        float i = input(0);
        float o = 1 / (1.0f + exp(i));

        output(0) = o;
    }
};

REGISTER_KERNEL_BUILDER(Name("SoftplusDerOp").Device(DEVICE_GPU), SoftplusDeri)


#endif // CNAI_DNN_DER_BASE_H
