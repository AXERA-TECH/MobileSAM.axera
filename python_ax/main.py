from sam_encoder import SAMEncoder
from sam_decoder import SAMDecoder
import cv2
import numpy as np
import argparse
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", "-i", type=str, default="../images/test.jpg", help="input image path")
    parser.add_argument("--output_dir", "-o", type=str, default="./output", help="result path")
    parser.add_argument("--chip", "-c", type=str, default="650", help="650 or 620E")
    args = parser.parse_args()
    
    encoder = SAMEncoder(f"../ax_model/mobile_sam_encoder_{args.chip}.axmodel")
    decoder = SAMDecoder(f"../ax_model/mobile_sam_decoder_{args.chip}.axmodel")
    
    image = cv2.imread(args.img_path)
    h, w, _ = image.shape
    image_embedding, scale = encoder.encode(image)
    
    # test.jpg
    point0 = (910, 641)
    point1 = (1488, 607)
    point2 = (579, 704)
    # truck.jpg
    # point0 = (500, 375)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, point in enumerate([point0, point1, point2]):
        image_draw = image.copy()
    
        output = decoder.decode(image_embedding[0], point = point,scale = scale)
        idx = output[0].argmax()
    
        image_draw = cv2.circle(image_draw, (int(point[0]), int(point[1])), 10, (0,255,0), -1)
        mask = output[1][:,idx,:,:][0]
        mask_mat = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        mask_mat[mask>0] = 255
        mask_mat = cv2.resize(mask_mat, (max(w, h),max(w, h)),interpolation=cv2.INTER_LINEAR)
        mask_mat = mask_mat[:h, :w]
        cv2.imwrite(f"{args.output_dir}/point_mask_point_{i}.jpg", mask_mat)
        mask_ovlap = np.zeros((mask_mat.shape[0], mask_mat.shape[1], 3), dtype=np.uint8)
        mask_ovlap[mask_mat>0] = [0, 255, 0]
        image_ovlap = cv2.addWeighted(image_draw, 1, mask_ovlap, 0.5, 0)
        cv2.imwrite(f"{args.output_dir}/point_mask_ovlap_point_{i}.jpg", image_ovlap)
    
    # for i in range(4):
    #     mask = output[1][:,i,:,:][0]
    #     mask_mat = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #     mask_mat[mask>0] = 255
    #     mask_mat = cv2.resize(mask_mat, (max(w, h),max(w, h)))
    #     mask_mat = mask_mat[:h, :w,:]
    #     cv2.imwrite(f"./output_ax/point_mask_{i}.jpg", mask_mat)
    
    # box: topleft x, topleft y, width, height
    # test.jpg
    box0 = (910 - 160, 641 - 430, 380, 940)
    box1 = (479, 482, 191, 518)
    box2 = (1345, 333, 289, 701)
    box3 = (1, 357, 311, 751)
    # truck.jpg
    # box0 = (1375, 550, 1650 - 1375, 800 - 550)
    # box1 = (75, 275, 1725 - 75, 850 - 275)
    # box2 = (425, 600, 700 - 425, 875 - 600)
    # box3 = (1240, 675, 1400 - 1240, 750 - 675)
    # car.jpg
    # box0 = (450, 170, 520 - 450, 350 - 170)
    # box1 = (350, 190, 450 - 350, 350 - 190)
    # box2 = (500, 170, 580 - 500, 350 - 170)
    # box3 = (580, 170, 640 - 580, 350 - 170)
    for i, box in enumerate([box0, box1, box2, box3]):
        image_draw = image.copy()
        output = decoder.decode(image_embedding[0], box = box,scale = scale)
        idx = output[0].argmax()

        image_draw = cv2.rectangle(image_draw, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (0,255,0), 2)
        # cv2.imwrite(f"{args.output_dir}/box_image_{i}.jpg", image)

        mask = output[1][:,idx,:,:][0]
        mask_mat = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        mask_mat[mask>0] = 255
        mask_mat = cv2.resize(mask_mat, (max(w, h),max(w, h)),interpolation=cv2.INTER_LINEAR)
        mask_mat = mask_mat[:h, :w]
        cv2.imwrite(f"{args.output_dir}/box_mask_box_{i}.jpg", mask_mat)
        mask_ovlap = np.zeros((mask_mat.shape[0], mask_mat.shape[1], 3), dtype=np.uint8)
        mask_ovlap[mask_mat>0] = [0, 255, 0]
        image_ovlap = cv2.addWeighted(image_draw, 1, mask_ovlap, 0.5, 0)
        cv2.imwrite(f"{args.output_dir}/box_mask_ovlap_box_{i}.jpg", image_ovlap)
    
    # for i in range(4):
    #     mask = output[1][:,i,:,:][0]
    #     mask_mat = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #     mask_mat[mask>0] = 255
    #     mask_mat = cv2.resize(mask_mat, (max(w, h),max(w, h)))
    #     mask_mat = mask_mat[:h, :w,:]
    #     cv2.imwrite(f"./output_ax/box_mask_{i}.jpg", mask_mat)
    