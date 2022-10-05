# model nsys profile
nsys profile -o logs/edsr_x2_profile --force-overwrite true \
   trtexec --loadEngine=model_zoo/EDSR_x2_dynamic_shape.plan \
   --warmUp=0 --duration=0 --iterations=500 --shapes=input:1x3x512x512
   