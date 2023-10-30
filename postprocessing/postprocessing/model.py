import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from torchvision import ops

class TritonPythonModel():

    def initialize(self, args):
        pass

    def get_bbox(self, data, threshold=0.25, nms_threshold=0.7):
        # data shape = (1, 84, 8400)
        data = data[0]
        n_classes = data.shape[0] - 4
        bboxes = []
        class_list = []
        score_list = []

        # Reshape l'array
        array = data.transpose(1, 0)
        boxes = array[:, :4]
        scores = array[:, 4:]

        for i in range(n_classes):
            idxs = np.where(scores[:, i] > threshold)[0]
            if len(idxs) > 0:
                class_scores = scores[idxs, i]
                class_boxes = boxes[idxs] # box format = x1, y1, w, h
                x1y1 = class_boxes[:, :2]
                wh = class_boxes[:, 2:]
                x2y2 = x1y1 + wh
                mod_class_boxes = torch.cat([x1y1, x2y2], dim = 1)
                keep = ops.nms(boxes=mod_class_boxes, scores=class_scores, iou_threshold=nms_threshold)
                for k in keep:
                    bboxes.append(class_boxes[k].numpy())
                    class_list.append(i)
                    score_list.append(class_scores[k].numpy())
                    #bboxes.append((class_boxes[k].tolist(), i, class_scores[k].tolist()))
        return np.array(bboxes), np.array(class_list), np.array(score_list)

    def execute(self, requests):
        responses = []
        for request in requests:

            input_data = pb_utils.get_input_tensor_by_name(request, "post_in")
            input = input_data.as_numpy()

            boxes, classes, scores = self.get_bbox(torch.tensor(input))

            boxes_tensor = pb_utils.Tensor("boxes", boxes)
            class_tensor = pb_utils.Tensor("classes", classes)
            score_tensor = pb_utils.Tensor("scores", scores)

            inference_response = pb_utils.InferenceResponse(output_tensors=[boxes_tensor, class_tensor, score_tensor])
            responses.append(inference_response)

        return responses
    