from pathlib import Path
import json
from PIL import Image

from config import OUTPUTS_DIR, DEVICE
from pipeline import FullInspectionPipeline
from utils import (
    get_next_run_dir,
    crop_box_from_image,
    draw_full_results,
    draw_crop_label,
    sanitize_name,
)


def main():
    image_path = Path("1.jpg").resolve()

    run_dir = get_next_run_dir(OUTPUTS_DIR, prefix="run_")
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    full_output_path = run_dir / "full_annotated.jpg"
    metadata_path = run_dir / "metadata.json"

    print(f"Input image : {image_path}")
    print(f"Run folder  : {run_dir}")
    print(f"Full image  : {full_output_path}")
    print(f"Crops dir   : {crops_dir}")

    pipeline = FullInspectionPipeline(device=DEVICE)

    outputs = pipeline.predict_image(
        image_path=image_path,
        det_conf=0.5,
        crop_pad=20,
        nest_iou_thresh=0.1,
    )

    print(f"\nFound {len(outputs)} final outputs")

    image = Image.open(image_path).convert("RGB")

    for i, out in enumerate(outputs, 1):
        print(f"\nDetection {i}")
        print(f"Component          : {out['component']}")
        print(f"BBox               : {out['bbox']}")
        print(f"Detector conf      : {out['det_conf']}")
        print(f"Classifier class   : {out.get('classifier_class')}")
        print(f"Classifier conf    : {out.get('classifier_conf')}")
        print(f"Top-k preds        : {out.get('topk_preds')}")
        print(f"Final class        : {out['final_class']}")
        print(f"Final conf         : {out['final_conf']}")
        print(f"Detector nest match: {out.get('detector_nest_match')}")

        crop = crop_box_from_image(image, out["bbox"], pad=20)
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
        crop.save(crops_dir / crop_filename)

    annotated = draw_full_results(image.copy(), outputs)
    annotated.save(full_output_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"\nSaved full annotated image to: {full_output_path}")
    print(f"Saved crop images to: {crops_dir}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()