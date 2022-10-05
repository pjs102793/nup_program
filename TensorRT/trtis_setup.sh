# model copy to triton_model_zoo/edsr_jit
cp model_zoo/EDSR_x2_dynamic_shape.pt \
   ../Triton_Inference_Server/trtis_model_zoo/edsr_jit/1/model.pt

# model copy to triton_model_zoo/edsr_trt
cp model_zoo/EDSR_x2_dynamic_shape.plan \
   ../Triton_Inference_Server/trtis_model_zoo/edsr_trt/1/model.plan
