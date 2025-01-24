from sam_encoder import SAMEncoder
from sam_decoder import SAMDecoder
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, default="models/mobile_sam_encoder.axmodel")
    args = parser.parse_args()
    
    encoder = SAMEncoder("models/mobile_sam_encoder.onnx")
    decoder = SAMDecoder("models/mobile_sam_decoder_low_res.onnx")
    
    image = cv2.imread(args.img_path)
    h, w, _ = image.shape
    image_embedding, scale = encoder.encode(image)
    
    point = (910, 641)
    
    
    output = decoder.decode(image_embedding[0], point = point,scale = scale)
    idx = output[0].argmax()
    
    mask = output[1][:,idx,:,:][0]
    mask_mat = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    mask_mat[mask>0] = 255
    mask_mat = cv2.resize(mask_mat, (max(w, h),max(w, h)),interpolation=cv2.INTER_LINEAR)
    mask_mat = mask_mat[:h, :w]
    cv2.imwrite("point_mask.jpg", mask_mat)
    mask_ovlap = np.zeros((mask_mat.shape[0], mask_mat.shape[1], 3), dtype=np.uint8)
    mask_ovlap[mask_mat>0] = [0, 255, 0]
    image_ovlap = cv2.addWeighted(image, 1, mask_ovlap, 0.5, 0)
    cv2.imwrite("point_mask_ovlap.jpg", image_ovlap)
    
    
    box = (910 - 160, 641 - 430, 380, 940)
    output = decoder.decode(image_embedding[0], box = box,scale = scale)
    idx = output[0].argmax()
    
    mask = output[1][:,idx,:,:][0]
    mask_mat = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    mask_mat[mask>0] = 255
    mask_mat = cv2.resize(mask_mat, (max(w, h),max(w, h)),interpolation=cv2.INTER_LINEAR)
    mask_mat = mask_mat[:h, :w]
    cv2.imwrite("box_mask.jpg", mask_mat)
    mask_ovlap = np.zeros((mask_mat.shape[0], mask_mat.shape[1], 3), dtype=np.uint8)
    mask_ovlap[mask_mat>0] = [0, 255, 0]
    image_ovlap = cv2.addWeighted(image, 1, mask_ovlap, 0.5, 0)
    cv2.imwrite("box_mask_ovlap.jpg", image_ovlap)
    