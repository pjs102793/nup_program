import sys
import cv2

import tritonclient
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm

from utils import preprocess, postprocess, logging_time


def set_shared_memory(
    triton_client, shm_ip0_handle, shm_op0_handle, image_data
):
    """
    Triton Inference Server에 등록할 Shared Memory를 OS 단에서 먼저 할당 해 준 후,
    Triton Inference Server에 키 값을 할당 해 줍니다.
    """
    input_byte_size = image_data.size * image_data.itemsize

    output_byte_size = input_byte_size * 4

    # Create Output0 Shared Memory and store shared memory handles
    shm_op0_handle.append(
        shm.create_shared_memory_region(
            "output0_data", "/output0_data", output_byte_size
        )
    )

    # Register Output0 shared memory with Triton Server
    triton_client.register_system_shared_memory(
        "output0_data", "/output0_data", output_byte_size
    )

    # Create Input0 Shared Memory and store shared memory handles
    shm_ip0_handle.append(
        shm.create_shared_memory_region(
            "input0_data", "/input0_data", input_byte_size
        )
    )

    # Register Input0 shared memory with Triton Server
    triton_client.register_system_shared_memory(
        "input0_data", "/input0_data", input_byte_size
    )

    # Put input data values into shared memory
    shm.set_shared_memory_region(shm_ip0_handle[0], [image_data])


def destroy_shared_memory(triton_client, shm_ip0_handle, shm_op0_handle):
    """
    Triton Inference Server에 등록된 Shared Memory 키 값을 삭제 후, OS단에 할당 되어있는 Shared
    Memory를 삭제합니다.
    """
    triton_client.unregister_system_shared_memory("output0_data")
    triton_client.unregister_system_shared_memory("input0_data")

    shm.destroy_shared_memory_region(shm_op0_handle[0])
    shm.destroy_shared_memory_region(shm_ip0_handle[0])


@logging_time
def infer(triton_client, input_data):
    input_byte_size = input_data.size * input_data.itemsize
    output_byte_size = input_byte_size * 4

    inputs = []
    outputs = []

    inputs.append(grpcclient.InferInput("input", input_data.shape, "FP32"))
    inputs[0].set_shared_memory("input0_data", input_byte_size)

    outputs.append(grpcclient.InferRequestedOutput("output"))
    outputs[0].set_shared_memory(f"output0_data", output_byte_size)

    # Inference
    results = triton_client.infer(
        model_name="SCUSR", inputs=inputs, outputs=outputs
    )

    # Get the output arrays from the results
    output0_data = results.get_output("output")

    return output0_data


def main():
    shm_ip0_handle = []
    shm_op0_handle = []

    # connect to server
    try:
        triton_client = grpcclient.InferenceServerClient("localhost:8011")
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    triton_client.unregister_system_shared_memory()

    # load image
    input0_data = cv2.imread("../TensorRT/EDSR-PyTorch/images/Lenna.png")

    input0_data = preprocess(input0_data)

    set_shared_memory(
        triton_client, shm_ip0_handle, shm_op0_handle, input0_data
    )

    output0 = infer(triton_client, input0_data)

    output0_data = shm.get_contents_as_numpy(
        shm_op0_handle[0],
        tritonclient.utils.triton_to_np_dtype(output0.datatype),
        output0.shape,
    )

    output0_data = postprocess(output0_data)

    cv2.imwrite("result/Lenna_SR.png", output0_data)

    destroy_shared_memory(triton_client, shm_ip0_handle, shm_op0_handle)

    print("Done!")


if __name__ == "__main__":
    main()
