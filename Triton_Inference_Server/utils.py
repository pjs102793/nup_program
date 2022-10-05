import time
import numpy as np


def logging_time(original_fn):
    """
    시간을 측정하는 함수
    """

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"Time [{original_fn.__name__}]: {end_time - start_time} sec")

        return result

    return wrapper_fn


@logging_time
def postprocess(result_img):
    """
    Triton Server에서 나온 Numpy array를 .
    """
    result_img = result_img.squeeze()
    result_img = np.transpose(result_img, (1, 2, 0))

    result_img = result_img.clip(0, 1)
    result_img *= 255

    result_img = np.ascontiguousarray(result_img, dtype=np.uint8)

    return result_img


@logging_time
def preprocess(input_img):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)
    input_img = input_img.astype(np.float32)
    input_img /= 255

    return input_img
