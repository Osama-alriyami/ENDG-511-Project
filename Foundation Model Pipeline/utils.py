# Utility functions used across the pipeline.

from PIL import Image
from torchvision import transforms


def crop_box_from_image(image, bbox, pad=0):
    x1, y1, x2, y2 = bbox
    w, h = image.size

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return image.crop((x1, y1, x2, y2))


def get_crop_transform(img_size=224):
     # crops bounding box with optional padding
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def compute_iou(box1, box2):
    # computes overlap between two boxes
    # used for matching nest detections
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, a2 - a1) * max(0, b2 - b1)

    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0

    return inter_area / union