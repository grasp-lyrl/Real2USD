import numpy as np
from ipdb import set_trace as st
import os
os.environ['YOLO_VERBOSE'] = 'False'
import cv2
import time
import torch

# Use CPU if no GPU or if GPU has unsupported CUDA capability (e.g. RTX 5090 sm_120)
def _infer_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        cap = torch.cuda.get_device_capability()
        # PyTorch pre-built binaries typically support up to sm_90; newer GPUs (e.g. sm_120) need a custom build
        if cap[0] > 9:
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"

_SEG_DEVICE = _infer_device()

from ultralytics import SAM, FastSAM, YOLO, YOLOE
from ultralytics import settings

"""
author: Christopher D. Hsu
email: chsu8@seas.upenn.edu
created: 11-25-2024

https://docs.ultralytics.com/models/yoloe/
pip install ultralytics
"""

class Segmentation:
    def __init__(self, model_path):
        model_name = os.path.basename(model_path)
        if model_name.startswith("FastSAM"):
            self.model = FastSAM(model_path)
        elif model_name.startswith("sam2.1"):
            self.model = SAM(model_path)
        else:
            self.model = YOLOE(model_path)
        self.model.to(_SEG_DEVICE)

        self.classes = ["a round table", "a church bench", "power outlet", "chair", "door", "a desk", "a sofa", "barstool", "file cabinet", "cocktail table", "side table", "elevator door"] #, "canopy bed", "office desk"]
        self.classes_usd = ["Table", "Chair", "Misc", "Chair", "Misc", "Table", "Chair", "Chair", "Storage", "Table", "Table", "Misc"]#, "Table", "Table"]

        self.track_id_counter = 999  # Counter for generating unique IDs

        self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))

    def crop_img_w_bbox(self, image, retina_masks=True, imgsz=1024, conf=0.8, iou=0.9):
        """
        given an image (cv2), segment image under params and crop the image with the bounding box
        conf : Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
        iou: Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
        disclaimer: segmentation tracks create a new track id if the object leaves the image

        returns:
        full image with annotations in cv2
        list of cropped images in numpy array
        list of cropped images in cv2
        list of bounding box points in numpy array
        list of segmentation mask points in numpy array
        list of segmentation tracks and object ids
        list of segmentation class names
        """
        # standard segementation
        # results = self.model(
        #     image, retina_masks=retina_masks, imgsz=imgsz, conf=conf, iou=iou, verbose=False
        # )

        # track segmentation
        # self.model.set_classes(self.classes)
        results = self.model.track(
            image, retina_masks=retina_masks, imgsz=imgsz, conf=conf, iou=iou, verbose=False, persist=True,
            device=_SEG_DEVICE,
        )
        # results is a list because you can feed in multiple images
        result = results[0]

        # Initialize empty lists for all outputs
        box_xyxy = []
        track_ids = []
        class_names = []
        img_arrs = []
        img_crop = []
        box_pts = []
        seg_pts = []
        mask_pts = []
        try:
            # First try to get the detections
            box_xyxy = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.cpu().numpy()
            
            # Try to get class indices and names
            try:
                class_indices = result.boxes.cls.cpu().numpy()
                class_names = [self.classes[int(idx)] for idx in class_indices]
                class_names_usd = [self.classes_usd[int(idx)] for idx in class_indices]
            except:
                # If no class indices, assign "" to all detections
                class_names = [""] * len(box_xyxy)
                class_names_usd = [""] * len(box_xyxy)

            # Then try to get track IDs separately
            track_ids = result.boxes.id.int().cpu().tolist()

        except:
            # If no detections at all, return empty lists
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), [], [], np.array([]), [], [], [], []

        image_arr = np.asarray(image)
        dim_x = image_arr.shape[1]
        dim_y = image_arr.shape[0]

        # crop images based on bounding box
        pad = 0
        for ii in range(len(box_xyxy)):
            # padded image crop and create a mask that we will use to segment point cloud
            seg_mask = np.zeros((dim_y, dim_x), dtype=bool)
            left_x = np.clip(box_xyxy[ii, 0]-pad, 0, dim_x-1).astype(int)
            right_x = np.clip(box_xyxy[ii, 2]+pad, 0, dim_x-1).astype(int)
            top_y = np.clip(box_xyxy[ii, 1]-pad, 0, dim_y-1).astype(int)
            bottom_y = np.clip(box_xyxy[ii, 3]+pad, 0, dim_y-1).astype(int)
            seg_mask[top_y:bottom_y, left_x:right_x] = True

            # grey background for image crop       
            left_x = box_xyxy[ii, 0].astype(int)
            right_x = box_xyxy[ii, 2].astype(int)
            top_y = box_xyxy[ii, 1].astype(int)
            bottom_y = box_xyxy[ii, 3].astype(int)
            center_xy = box_xyxy[ii, 0:2] + (box_xyxy[ii, 2:4] - box_xyxy[ii, 0:2]) / 2
            # [u,v,3]
            cropped_img_arr = image_arr[top_y:bottom_y, left_x:right_x, :]
            img_arrs.append(cropped_img_arr)

            # use the full bounding box image
            # cropped_img_arr = cv2.copyMakeBorder(
            #     cropped_img_arr,
            #     pad,
            #     pad,
            #     pad,
            #     pad,
            #     cv2.BORDER_CONSTANT,
            #     value=[128, 128, 128],
            # )

            # for masked cropped image, first create a grey background add mask data and then crop
            grey = np.zeros((dim_y, dim_x, 3), dtype=np.uint8)
            grey[:] = [128, 128, 128]
            grey[masks.data[ii]>0] = image_arr[masks.data[ii]>0]
            pad = 10
            left_x = np.clip(box_xyxy[ii, 0]-pad, 0, dim_x-1).astype(int)
            right_x = np.clip(box_xyxy[ii, 2]+pad, 0, dim_x-1).astype(int)
            top_y = np.clip(box_xyxy[ii, 1]-pad, 0, dim_y-1).astype(int)
            bottom_y = np.clip(box_xyxy[ii, 3]+pad, 0, dim_y-1).astype(int)
            cropped_img_arr = grey[top_y:bottom_y, left_x:right_x, :]

            img_crop.append(cv2.cvtColor(cropped_img_arr, cv2.COLOR_RGB2BGR))
            box_pts.append(
                np.array(
                    [
                        [center_xy[0], center_xy[1]],
                        [box_xyxy[ii, 0], box_xyxy[ii, 1]],
                        [box_xyxy[ii, 0], box_xyxy[ii, 3]],
                        [box_xyxy[ii, 2], box_xyxy[ii, 1]],
                        [box_xyxy[ii, 2], box_xyxy[ii, 3]],
                    ]
                )
            )


            # get the indeces of the image where the mask is true
            rows, cols = np.where(masks.data[ii])
            mask_pts.append(np.column_stack((cols, rows)))

            # rows, cols = np.where(seg_mask)
            # seg_pts.append(np.column_stack((cols, rows)))
        # convert annotated image to cv2
        full_img = result.plot(labels=True, masks=True, probs=False, conf=True)
        full_img_cv2 = cv2.cvtColor(np.asarray(full_img), cv2.COLOR_RGB2BGR)

        """
        full image with annotations in cv2
        list of cropped images in numpy array
        list of cropped images in cv2
        list of bounding box points in numpy array
        list of segmentation mask points in numpy array
        list of segmentation tracks and object ids
        list of segmentation class names
        """
        return full_img_cv2, img_arrs, img_crop, np.asarray(box_pts), mask_pts, track_ids, class_names, class_names_usd


if __name__ == "__main__":
    # model_path = "models/yoloe-11l-seg.pt"
    model_path = "models/yoloe-11l-seg-pf.pt"
    # if using prompt free, pf, you need to comment out .set_classes() in line 36
    sam = Segmentation(model_path)
    image_path = (
        "/data/tests/test5.png"
    )
    # Load the PNG image into cv2 format
    image = cv2.imread(image_path)
    result, img_arrs, img_pils, box_pts, mask_pts, track_ids, class_names, class_names_usd = sam.crop_img_w_bbox(image, conf=0.5, iou=0.2)


    # convert result from cv2 to PIL
    import PIL
    result_pil = PIL.Image.fromarray(result)

    # save the result
    result_pil.save("result.png")