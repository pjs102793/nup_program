import sys
import cv2

import tritonclient.grpc as grpcclient

from utils import preprocess, postprocess, logging_time


@logging_time
def infer(triton_client, input_data):
    inputs = []
    outputs = []

    inputs.append(grpcclient.InferInput("input", input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)

    outputs.append(grpcclient.InferRequestedOutput("output"))

    # Inference
    results = triton_client.infer(
        model_name="edsr_trt", inputs=inputs, outputs=outputs
    )

    # Get the output arrays from the results
    output0_data = results.as_numpy("output")

    return output0_data


def main():
    # connect to server
    try:
        triton_client = grpcclient.InferenceServerClient("localhost:8011")
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    # load image
    input0_data = cv2.imread("../TensorRT/EDSR-PyTorch/images/Lenna.png")
    input0_data = preprocess(input0_data)

    output0_data = infer(triton_client, input0_data)

    output0_data = postprocess(output0_data)

    cv2.imwrite("result/Lenna_SR.png", output0_data)

    print("Done!")


if __name__ == "__main__":
    main()
