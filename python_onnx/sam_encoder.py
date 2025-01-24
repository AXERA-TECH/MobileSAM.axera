import onnxruntime
import cv2
import numpy as np

class SAMEncoder:
    def __init__(self,model_path):
        self.sess = onnxruntime.InferenceSession(model_path)
        self.input_shape = (1024, 1024)
    
    def letterbox(self, image, target_size, color=(114, 114, 114)):
        """
        将图像调整为目标大小，同时保持原始长宽比，并填充空白区域。
        
        :param image: 输入图像 (H, W, C)
        :param target_size: 目标尺寸 (width, height)
        :param color: 填充颜色 (B, G, R)
        :return: 调整后的图像，缩放比例，填充区域
        """
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size

        # 计算缩放比例
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 调整图像大小
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 计算填充
        pad_width = (target_width - new_width) // 2
        pad_height = (target_height - new_height) // 2

        # 填充图像
        padded_image = cv2.copyMakeBorder(
            resized_image, 
            0 , target_height - new_height , 
            0, target_width - new_width ,
            cv2.BORDER_CONSTANT, 
            value=color
        )

        return padded_image, scale, (pad_width, pad_height)
    
    def preprocess(self,image):
        padded_image, scale, (pad_width, pad_height) = self.letterbox(image, self.input_shape)
        
        padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        return padded_image, scale
        
    def encode(self,image):
        padded_image, scale = self.preprocess(image)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
        padded_image = (padded_image.astype(np.float32)/255 - mean)/std
        
        padded_image = np.transpose(padded_image, (2, 0, 1))
        padded_image = np.expand_dims(padded_image, axis=0)
        

        return self.sess.run(None,{self.sess.get_inputs()[0].name:padded_image}), scale