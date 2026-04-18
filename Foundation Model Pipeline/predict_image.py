# Runs pipeline on a single image.
# Saves annotated image + cropped detections.
# Also measures inference time.
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time
import torch

from pipeline import FullInspectionPipeline


def get_font_for_image(image, scale=0.035):
    """
    scale: percentage of image height
    """
    img_w, img_h = image.size
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


def get_next_run_dir(base_dir="outputs"):
    base = Path(base_dir)
    base.mkdir(exist_ok=True)

    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]

    if len(existing) == 0:
        run_id = 1
    else:
        run_ids = [int(d.name.split("_")[1]) for d in existing]
        run_id = max(run_ids) + 1

    run_dir = base / f"run_{run_id:03d}"
    run_dir.mkdir()

    return run_dir


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
        text_x = x1
        pad = int(image.size[1] * 0.01)
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

    draw.rectangle(
        [0, 0, text_w + 10, text_h + 8],
        fill=fill_color
    )
    draw.text((5, 4), label, fill="white", font=font)

    return crop_image


def crop_with_padding(image, bbox, pad=20):
    x1, y1, x2, y2 = bbox
    w, h = image.size

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return image.crop((x1, y1, x2, y2))


def sanitize_name(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def sync_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_pipeline(pipeline, image_path, det_conf=0.5, crop_pad=20, nest_iou_thresh=0.1,
                  warmup_runs=5, timed_runs=30):
    """
    Measure only pipeline.predict_image() time.
    Keeps timing separate from drawing and saving.
    """

     # warmup runs to stabilize GPU timing
    print(f"\nWarming up for {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = pipeline.predict_image(
            image_path=image_path,
            det_conf=det_conf,
            crop_pad=crop_pad,
            nest_iou_thresh=nest_iou_thresh
        )

    times_ms = []

    print(f"Timing inference over {timed_runs} runs...")
    for _ in range(timed_runs):
        sync_if_cuda(pipeline.device)
        start_time = time.perf_counter()

        _ = pipeline.predict_image(
            image_path=image_path,
            det_conf=det_conf,
            crop_pad=crop_pad,
            nest_iou_thresh=nest_iou_thresh
        )

        sync_if_cuda(pipeline.device)
        end_time = time.perf_counter()

        times_ms.append((end_time - start_time) * 1000.0)

    avg_time = sum(times_ms) / len(times_ms)
    min_time = min(times_ms)
    max_time = max(times_ms)

    return avg_time, min_time, max_time, times_ms



image_path = Path("11.jpg").resolve() # Input image
run_dir = get_next_run_dir("outputs")

crops_dir = run_dir / "crops"
crops_dir.mkdir(parents=True, exist_ok=True)

full_output_path = run_dir / "full_annotated.jpg"

print(f"Input image : {image_path}")
print(f"Run folder  : {run_dir}")
print(f"Full image  : {full_output_path}")
print(f"Crops dir   : {crops_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device      : {device}")

# create pipeline
pipeline = FullInspectionPipeline(device=device)

# More accurate timing
avg_time, min_time, max_time, times_ms = time_pipeline(
    pipeline=pipeline,
    image_path=image_path,
    det_conf=0.5,
    crop_pad=20,
    nest_iou_thresh=0.1,
    warmup_runs=5,
    timed_runs=30
)

print(f"\nAverage inference time: {avg_time:.2f} ms")
print(f"Min inference time    : {min_time:.2f} ms")
print(f"Max inference time    : {max_time:.2f} ms")

# Run once more for the actual saved outputs
sync_if_cuda(device)
start_time = time.perf_counter()

# run one final inference for saving results
outputs = pipeline.predict_image(
    image_path=image_path,
    det_conf=0.5,
    crop_pad=20,
    nest_iou_thresh=0.1
)

sync_if_cuda(device)
end_time = time.perf_counter()
single_run_time = (end_time - start_time) * 1000.0

print(f"\nSingle saved run inference time: {single_run_time:.2f} ms")
print(f"Found {len(outputs)} final outputs")

image = Image.open(image_path).convert("RGB")

per_detection_time = single_run_time / max(len(outputs), 1)

for i, out in enumerate(outputs, 1):
    print(f"\nDetection {i}")
    print(f"Component          : {out['component']}")
    print(f"BBox               : {out['bbox']}")
    print(f"Detector conf      : {out['det_conf']}")
    print(f"Classifier class   : {out.get('classifier_class')}")
    print(f"Classifier conf    : {out.get('classifier_conf')}")
    print(f"Final class        : {out['final_class']}")
    print(f"Final conf         : {out['final_conf']}")
    print(f"Detector nest match: {out.get('detector_nest_match')}")
    print(f"Approx time share  : {per_detection_time:.2f} ms")

    crop = crop_with_padding(image, out["bbox"], pad=20)
    crop = draw_crop_label(
        crop_image=crop,
        component=out["component"],
        final_class=out["final_class"],
        final_conf=out["final_conf"]
    )

    crop_filename = (
        f"{i:03d}_"
        f"{sanitize_name(out['component'])}_"
        f"{sanitize_name(out['final_class'])}.jpg"
    )
    crop_path = crops_dir / crop_filename
    crop.save(crop_path)

full_annotated = draw_full_results(image.copy(), outputs)
full_annotated.save(full_output_path)

print(f"\nSaved full annotated image to: {full_output_path}")
print(f"Saved crop images to: {crops_dir}")

