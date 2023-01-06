# onnx model exporting
python3 test.py \
        --weights-file ../model_zoo/EDSR_x2_patch240_nonoise_best.pth \
        --image-file images/Lenna.png --scale 2 --onnx-export

# jit model exporting
# python3 test.py --weights-file ../model_zoo/EDSR_x2_patch240_nonoise_best.pth --image-file images/Lenna.png --scale 2 --jit-export

# onnx and jit model exporting
# python3 test.py \
#         --weights-file ../model_zoo/EDSR_x2_patch240_nonoise_best.pth \
#         --image-file images/Lenna.png --scale 2 --onnx-export --jit-export

# tensorrt model exporting
trtexec --onnx=../model_zoo/EDSR_x2_dynamic_shape.onnx --explicitBatch --device=1 --fp16 \
        --workspace=18000 --saveEngine=../model_zoo/EDSR_x2_dynamic_shape.plan \
        --minShapes=input:1x3x1x1 --optShapes=input:1x3x1000x1000 \
        --maxShapes=input:1x3x2000x2000 --verbose \
        --profilingVerbosity=detailed --dumpLayerInfo --dumpProfile
