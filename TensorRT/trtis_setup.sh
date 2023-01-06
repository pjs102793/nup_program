# model copy to triton_model_zoo/edsr_jit
# cp model_zoo/EDSR_x2_dynamic_shape.pt \
#    ../Triton_Inference_Server/trtis_model_zoo/edsr_jit/1/model.pt

# model copy to triton_model_zoo/edsr_trt
mkdir ../Triton_Inference_Server/trtis2_model_zoo/edsr/1

cp model_zoo/EDSR_x2_dynamic_shape.plan \
   ../Triton_Inference_Server/trtis2_model_zoo/edsr/1/model.plan
