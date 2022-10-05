"""
Image Inference Client of Triton Inference Server
"""

import cv2
import numpy as np
import sys
from functools import partial
import os

import tritonclient
import tritonclient.grpc as grpcclient
from tritonclient import utils
from tritonclient.utils import InferenceServerException
import tritonclient.utils.shared_memory as shm

import queue

from utils import preprocess, postprocess


class UserData:
    def __init__(self):
        self.completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    user_data.completed_requests.put((result, error))


def request_generator(
    triton_client, batched_image_data, shm_ip0_handle, shm_op0_handle
):

    input_byte_size = batched_image_data.size * batched_image_data.itemsize
    output_byte_size = input_byte_size * 4

    # Create Output0 Shared Memory and store shared memory handles
    shm_op0_handle.append(
        shm.create_shared_memory_region(
            f"output0_data", f"/output0_data", output_byte_size
        )
    )

    # Register Output0 and Output1 shared memory with Triton Server
    triton_client.register_system_shared_memory(
        f"output0_data", f"/output0_data", output_byte_size
    )

    # Create Input0 in Shared Memory and store shared memory handles
    shm_ip0_handle.append(
        shm.create_shared_memory_region(
            f"input0_data", f"/input0_data", input_byte_size
        )
    )

    # Register Input0 shared memory with Triton Server
    triton_client.register_system_shared_memory(
        f"input0_data", f"/input0_data", input_byte_size
    )

    # Put input data values into shared memory
    shm.set_shared_memory_region(shm_ip0_handle[0], [batched_image_data])

    # Set the input data
    inputs = []
    inputs.append(
        grpcclient.InferInput("input", batched_image_data.shape, "FP32")
    )
    inputs[0].set_shared_memory(f"input0_data", input_byte_size)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output"))
    outputs[0].set_shared_memory(f"output0_data", output_byte_size)

    yield inputs, outputs


def main():
    # Create gRPC client for communicating with the server
    triton_client = grpcclient.InferenceServerClient(
        url="localhost:8011", verbose=False
    )
    triton_client.unregister_system_shared_memory()

    # Preprocess the images into input data according to model
    input0_data = cv2.imread("../TensorRT/EDSR-PyTorch/images/Lenna.png")
    input0_data = preprocess(input0_data)

    shm_ip0_handle = []
    shm_op0_handle = []

    output0_data = []

    user_data = UserData()

    triton_client.start_stream(partial(completion_callback, user_data))

    # Send request
    try:
        for inputs, outputs in request_generator(
            triton_client, input0_data, shm_ip0_handle, shm_op0_handle
        ):
            triton_client.async_stream_infer(
                "edsr_trt", inputs, outputs=outputs
            )

    except InferenceServerException as e:
        print("inference failed: " + str(e))
        triton_client.stop_stream()

        return

    triton_client.stop_stream()

    results, error = user_data.completed_requests.get()

    if error is not None:
        print(f"inference failed: {error}")

        triton_client.unregister_system_shared_memory(f"input0_data")
        triton_client.unregister_system_shared_memory(f"output0_data")

        shm.destroy_shared_memory_region(shm_ip0_handle[0])
        shm.destroy_shared_memory_region(shm_op0_handle[0])

    output0 = results.get_output("output")

    output0_data = shm.get_contents_as_numpy(
        shm_op0_handle[0],
        tritonclient.utils.triton_to_np_dtype(output0.datatype),
        output0.shape,
    )

    result_img = postprocess(output0_data)

    cv2.imwrite("result/Lenna_SR.png", result_img)

    triton_client.unregister_system_shared_memory(f"input0_data")
    triton_client.unregister_system_shared_memory(f"output0_data")

    shm.destroy_shared_memory_region(shm_ip0_handle[0])
    shm.destroy_shared_memory_region(shm_op0_handle[0])

    print("done")


if __name__ == "__main__":
    main()
