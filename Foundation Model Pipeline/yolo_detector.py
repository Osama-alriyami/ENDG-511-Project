# Simple wrapper around YOLO model.
# Converts YOLO output into clean dictionary format.

from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path, device="cuda"):
        # load YOLO model
        self.model = YOLO(str(model_path))
        self.device = device

    def predict(self, image_path, conf=0.25): # change conf later
          # run YOLO inference
        results = self.model.predict(source=str(image_path), conf=conf, device=self.device, verbose=False)

        detections = []
        for r in results:
            names = r.names
            boxes = r.boxes

            if boxes is None:
                continue

            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                # class id → class name
                cls_id = int(b.cls[0].item())
                 # confidence score
                det_conf = float(b.conf[0].item())

                detections.append({
                    "bbox": [int(v) for v in xyxy],
                    "component": names[cls_id],
                    "det_conf": det_conf
                })

        return detections