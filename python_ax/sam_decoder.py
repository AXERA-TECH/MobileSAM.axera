import onnxruntime
import cv2
import numpy as np


class SAMDecoder:

    def __init__(self, model_path):
        self.sess = onnxruntime.InferenceSession(model_path)

        
        self.mask = np.zeros((1, 1, 256, 256), np.float32)
        self.has_mask = np.array([0], np.float32)

    def decode(self, image_embedding, point = None, box = None, scale = None):
        if point is not None:
            point = np.array(point).astype(np.float32) * scale
            point_coords = np.array([point, (0,0), (0,0), (0,0), (0,0)]).astype(np.float32).reshape((1, -1, 2))
            point_labels = np.array([1, 0, 0, 0, 0], np.float32).reshape((1, -1))
        elif box is not None:
            box = np.array(box).astype(np.float32)*scale
            x, y, w, h = box
            center = np.array([x + w/2, y + h/2], np.float32)
            topleft = np.array([x, y], np.float32)
            bottomright = np.array([x + w, y + h], np.float32)
            point_coords = np.array([center, topleft, bottomright, (0,0), (0,0)]).astype(np.float32).reshape((1, -1, 2))
            point_labels = np.array([1, 2, 3, 0, 0], np.float32).reshape((1, -1))
        else:
            raise ValueError("Either point or box must be provided.")
        inputs = {
            "image_embeddings": image_embedding,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": self.mask,
            "has_mask_input": self.has_mask,
        }
        outputs = self.sess.run(None, inputs)
        return outputs
