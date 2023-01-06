# Nup Tiering Program 실습 코드

Nup Tiering Program에서 세미나 할 때 사용한 실습 코드입니다.

실습의 목표는 EDSR 모델을 TensorRT 모델로 생성하고 Triton Inference Server(TRTIS)로 배포합니다.

이 실습은 Docker 이미지인 nvcr.io/nvidia/tensorrt:22.04-py3 이미지에 Pytorch는 1.12.0+cu116 버전을 사용하였습니다. 자세한 것은 Dockerfile을 참고하시기 바랍니다.

빌드 방법은 다음과 같습니다.
```bash
make build
```

빌드된 이미지에 접속하는 명령어는 다음과 같습니다.
```bash
make run
```

아래 Easy TRTIS Setup와 Run Client 부분은 Dockerfile로 빌드된 이미지를 사용하여 작업하는 것을 권장합니다.
## Easy TRTIS Setup
```bash
# Step 1: pth to jit and TensorRT Model
cd /workspace/test/TensorRT/EDSR-PyTorch

bash ./exporting_run.sh

# Step 2: model copy to trtis_model_zoo
cd /workspace/test/TensorRT

bash trtis_setup.sh
```
이 작업을 하시게 되면 EDSR pth 모델을 jit, onnx, tensorrt model을 생성 후 jit, tensorrt model을 trtis_model_zoo로 옮깁니다. 그 후 컨테이너를 나와서 아래 명령어를 입력합니다. 

--ipc==host를 해주는 이유는 공유 메모리를 사용하기 위해서 설정합니다. Triton Server에서 Docker를 사용하실 때도 --ipc=host가 되어 있어야하고, Triton Client에서 Docker를 사용할 때도 --ipc=host가 되어 있어야합니다.

```bash
# Local Machine 에서 가동
docker run --rm --name="trtis2" --gpus='"device=3"' -p 8010:8000 -p 8011:8001 -p 8012:8002 --ipc=host -v /full/path/to/nup_program/Triton_Inference_Server/trtis_model_zoo:/models nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --model-repository=/models
```

## Run client
```bash
cd /workspace/test/Triton_Inference_Server
python infer_client.py

# 아래가 출력되면 성공!
Time [preprocess]: 0.0011217594146728516 sec
Time [infer]: 0.06110072135925293 sec
Time [postprocess]: 0.012495279312133789 sec
Done!
```

## Shell
각 폴더 내에 쉘 스크립팅이 있습니다. 해당 스크립트는 편의를 위해 상대경로로 작성되어 있으니, 쉘 스크립트를 실행 하실 때 해당 경로로 이동 후 직접 실행하시기 바랍니다.

TensorRT 디렉토리에는 nsight system 로그를 확인할 수 있는 nsys_profile.sh가 있고, trtis_setup.sh를 통해서 export했던 모델들을 Triton_Inference_Server에 trtis_model_zoo에 한번에 모델을 옮길 수 있는 cp 명령어가 저장되어 있습니다.

TensorRT/EDSR-PyTorch 디렉토리에는 onnx, jit 모델을 생성하고 TensorRT 모델을 생성하는 커맨드가 들어있습니다.

Triton_Inference_Server 디렉토리에는 perf_analyzer_run.sh에는 Triton_Inference_Server에 올려져있는 모델들의 성능 체크를 할수 있는 쉘 스크립트입니다.

## 기타 참고 사이트
TensorRT Document : https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html

TensorRT Tools (PTQ, QAT, trt_explorer ..) : https://github.com/NVIDIA/TensorRT/tree/main/tools

TensorRT Sparsity : https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity

TensorRT Custom Layer : https://github.com/wang-xinyu/tensorrtx

Triton Inference Server Document : https://github.com/triton-inference-server/server/tree/main/docs

Nsight Systems Document : https://docs.nvidia.com/nsight-systems/UserGuide/index.html
