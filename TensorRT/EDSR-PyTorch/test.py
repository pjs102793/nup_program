import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
import tqdm

from models import EDSR
from utils import preprocess, postprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--num-feats", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--res-scale", type=float, default=1.0)
    parser.add_argument("--onnx-export", action="store_true")
    parser.add_argument("--jit-export", action="store_true")
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EDSR(
        scale_factor=args.scale,
        num_channels=args.num_channels,
        num_feats=args.num_feats,
        num_blocks=args.num_blocks,
        res_scale=args.res_scale,
    ).to(device)
    try:
        model.load_state_dict(
            torch.load(args.weights_file, map_location=device)
        )
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(
            args.weights_file, map_location=lambda storage, loc: storage
        ).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert("RGB")

    image = preprocess(image).to(device)

    with torch.inference_mode():
        # [1, 3, 512, 512]
        preds = model(image)

        if args.onnx_export:
            print("onnx exporting...")
            torch.onnx.export(
                model,  # 실행될 모델
                image,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                "../model_zoo/EDSR_x2_dynamic_shape.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                opset_version=14,  # 모델을 변환할 때 사용할 ONNX 버전
                do_constant_folding=True,  # 최적화 시 상수 폴딩을 사용할지의 여부
                input_names=["input"],  # 모델의 입력값을 가리키는 이름
                output_names=["output"],  # 모델의 출력값을 가리키는 이름
                dynamic_axes={  # 가변적인 길이를 가진 차원
                    "input": {
                        2: "input_height",
                        3: "input_width",
                    },
                    "output": {
                        2: "output_height",
                        3: "output_width",
                    },
                },
            )

        if args.jit_export:
            traced_model = torch.jit.trace(model, image)
            torch.jit.save(
                traced_model, "../model_zoo/EDSR_x2_dynamic_shape.pt"
            )
        print(preds.size())

    output = postprocess(preds)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace(".", "_EDSR_x{}.".format(args.scale)))
