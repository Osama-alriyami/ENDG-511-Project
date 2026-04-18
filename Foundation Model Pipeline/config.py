from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo" / "best.pt"
ENCODER_PATH = BASE_DIR / "models" / "encoder" / "pruned_finetuned_encoder.pt"
HEADS_DIR = BASE_DIR / "models" / "heads"

DEVICE = "cuda"

IMG_SIZE = 224

COMPONENTS = ["damper", "fitting", "insulator", "plate", "spacer"]

HEAD_FILES = {
    "damper": HEADS_DIR / "damper_classifier.pt",
    "fitting": HEADS_DIR / "fitting_classifier.pt",
    "insulator": HEADS_DIR / "insulator_classifier.pt",
    "plate": HEADS_DIR / "plate_classifier.pt",
    "spacer": HEADS_DIR / "spacer_classifier.pt",
}