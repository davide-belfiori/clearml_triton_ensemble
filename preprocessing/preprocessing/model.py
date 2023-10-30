import triton_python_backend_utils as pb_utils
import numpy as np
from PIL import Image

class TritonPythonModel():

    def initialize(self, args):
        self.target_size = (640, 640)

    def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """
            Expected input data of shape (<img_height>, <img_width>, 3) and INT8 dtype
        """
        image = Image.fromarray(input_data)
        image = image.resize(self.target_size)
        processed_data = np.array(image)
        processed_data = processed_data.transpose((2, 0, 1))
        processed_data = processed_data.astype(np.float32)
        processed_data = processed_data / 255
        processed_data = processed_data[None, ...]
        return processed_data

    def execute(self, requests):
        responses = []
        for request in requests:

            input_data = pb_utils.get_input_tensor_by_name(request, "pre_in")
            input_array: np.ndarray = input_data.as_numpy()
            print("PREPROCESSING: input data shape =", input_array.shape)
            print("PREPROCESSING: input data type =", input_array.dtype)
            
            output_array = self.process_input(input_array)
            print("PREPROCESSING: processed data shape =", output_array.shape)
            print("PREPROCESSING: processed data type =", output_array.dtype)
            
            output_tensor = pb_utils.Tensor("pre_out", output_array)

            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
