from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo" / "best.pt"
ENCODER_PATH = BASE_DIR / "models" / "encoder" / "encoder_only.pt"
PROTONET_DIR = BASE_DIR / "models" / "protonet"

TRAIN_DATA_ROOT = BASE_DIR / "taskhead_classifier_dataset"

OUTPUTS_DIR = BASE_DIR / "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

COMPONENT_NAMES = {"damper", "fitting", "insulator", "plate"}
SPECIAL_DEFECT_NAMES = {"nest"}

PROTONET_FILES = {
    "damper": PROTONET_DIR / "damper_protonet.pt",
    "fitting": PROTONET_DIR / "fitting_protonet.pt",
    "insulator": PROTONET_DIR / "insulator_protonet.pt",
    "plate": PROTONET_DIR / "plate_protonet.pt",
    "spacer": PROTONET_DIR / "spacer_protonet.pt",
}