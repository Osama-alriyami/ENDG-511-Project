from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def get_next_run_dir(base_dir="outputs", prefix="run_"):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)]

    if len(existing) == 0:
        run_id = 1
    else:
        ids = []
        for d in existing:
            try:
                ids.append(int(d.name.replace(prefix, "")))
            except Exception:
                pass
        run_id = max(ids) + 1 if len(ids) > 0 else 1

    run_dir = base / f"{prefix}{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def crop_box_from_image(image, bbox, pad=20):
    x1, y1, x2, y2 = bbox
    w, h = image.size

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return image.crop((x1, y1, x2, y2))


def get_crop_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def compute_iou(box1, box2):
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


def sanitize_name(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def get_font_for_image(image, scale=0.035):
    _, img_h = image.size
    font_size = int(img_h * scale)
    font_size = max(12, min(font_size, 60))

    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return (len(text) * 7, 16)


def draw_full_results(image, outputs):
    draw = ImageDraw.Draw(image)
    font = get_font_for_image(image, scale=0.035)

    for out in outputs:
        x1, y1, x2, y2 = out["bbox"]
        component = out["component"]
        final_class = out["final_class"]
        final_conf = out["final_conf"]

        label = f"{component} | {final_class} | {final_conf:.2f}"

        if str(final_class).lower() == "good":
            box_color = (30, 144, 255)
            fill_color = (30, 144, 255)
        else:
            box_color = (220, 20, 60)
            fill_color = (220, 20, 60)

        box_thickness = max(2, int(image.size[1] * 0.004))
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_thickness)

        text_w, text_h = get_text_size(draw, label, font)
        pad = max(6, int(image.size[1] * 0.01))

        text_x = x1
        text_y = max(0, y1 - text_h - pad)

        draw.rectangle(
            [text_x, text_y, text_x + text_w + 10, text_y + text_h + 8],
            fill=fill_color
        )
        draw.text((text_x + 5, text_y + 4), label, fill="white", font=font)

    return image


def draw_crop_label(crop_image, component, final_class, final_conf):
    draw = ImageDraw.Draw(crop_image)
    font = get_font_for_image(crop_image, scale=0.08)

    label = f"{component} | {final_class} | {final_conf:.2f}"

    if str(final_class).lower() == "good":
        fill_color = (30, 144, 255)
    else:
        fill_color = (220, 20, 60)

    text_w, text_h = get_text_size(draw, label, font)

    draw.rectangle([0, 0, text_w + 10, text_h + 8], fill=fill_color)
    draw.text((5, 4), label, fill="white", font=font)

    return crop_image