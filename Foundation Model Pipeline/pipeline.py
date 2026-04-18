# This file defines the full pipeline:
# YOLO detection → crop → classification → final decision
# It combines detector + classifier heads together.
from PIL import Image
import torch

from config import YOLO_MODEL_PATH, DEVICE, IMG_SIZE
from yolo_detector import YOLODetector
from component_heads import HeadManager
from utils import crop_box_from_image, get_crop_transform, compute_iou


class FullInspectionPipeline:
    def __init__(self, device=DEVICE):
        self.device = device

        # YOLO detector for finding components
        self.detector = YOLODetector(YOLO_MODEL_PATH, device=device)

        # classification heads for each component
        self.head_manager = HeadManager(device=device)

        # transform applied to cropped regions
        self.crop_tfm = get_crop_transform(IMG_SIZE)

        self.component_names = {"damper", "fitting", "insulator", "plate"}
        self.special_defect_names = {"nest"}

    def predict_image(self, image_path, det_conf=0.1, crop_pad=20, nest_iou_thresh=0.1):
        # load image
        image = Image.open(image_path).convert("RGB")

        # run YOLO detection
        detections = self.detector.predict(image_path, conf=det_conf)

        component_dets = []
        nest_dets = []

        for det in detections:
            label = det["component"].lower()

            if label in self.special_defect_names:
                nest_dets.append(det)
            elif label in self.component_names:
                component_dets.append(det)

        outputs = []

        # separate normal components and nest detections
        for det in component_dets:
            component = det["component"]

             # skip if no classifier exists
            if component not in self.head_manager.models:
                continue


            # crop detected region    
            crop = crop_box_from_image(image, det["bbox"], pad=crop_pad)

            # convert to tensor
            crop_tensor = self.crop_tfm(crop).unsqueeze(0).to(self.device)
            
            # run classifier
            cls_out = self.head_manager.predict(component, crop_tensor)
            classifier_label = cls_out["pred_class"].lower()

            detector_says_nest = False
            matched_nest_conf = None

            for nest_det in nest_dets:
                iou = compute_iou(det["bbox"], nest_det["bbox"])
                if iou >= nest_iou_thresh:
                    detector_says_nest = True
                    matched_nest_conf = nest_det["det_conf"]
                    break

             # if either detector OR classifier says "nest", final result = nest
            if detector_says_nest or classifier_label == "nest":
                final_class = "nest"
                final_conf = max(
                    matched_nest_conf if matched_nest_conf is not None else 0.0,
                    cls_out["cls_conf"] if classifier_label == "nest" else 0.0
                )
            else:
                final_class = cls_out["pred_class"]
                final_conf = cls_out["cls_conf"]

            outputs.append({
                "component": component,
                "bbox": det["bbox"],
                "det_conf": det["det_conf"],
                "classifier_class": cls_out["pred_class"],
                "classifier_conf": cls_out["cls_conf"],
                "final_class": final_class,
                "final_conf": final_conf,
                "detector_nest_match": detector_says_nest,
            })

        # Optional, also return raw nest detections that did not overlap components
        for nest_det in nest_dets:
            overlaps_any = any(
                compute_iou(nest_det["bbox"], comp_det["bbox"]) >= nest_iou_thresh
                for comp_det in component_dets
            )

            if not overlaps_any:
                outputs.append({
                    "component": "unknown_component",
                    "bbox": nest_det["bbox"],
                    "det_conf": nest_det["det_conf"],
                    "classifier_class": None,
                    "classifier_conf": None,
                    "final_class": "nest",
                    "final_conf": nest_det["det_conf"],
                    "detector_nest_match": True,
                })

        return outputs