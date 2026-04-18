from ultralytics import YOLO
import torch
import os


# =========================================================
# SETTINGS
# =========================================================
DATA_YAML = "D:\ENDG 511 Project\yolo_detection_plus_plad/data.yaml"    
MODEL_NAME = "yolov8s.pt"                  
PROJECT_NAME = "yolo_runs"
RUN_NAME = "yolo_component_detector"

EPOCHS = 80
IMG_SIZE = 960
BATCH_SIZE = 4
WORKERS = 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"

PATIENCE = 20
SAVE_PERIOD = 10


# =========================================================
# MAIN TRAINING
# =========================================================
def main():
    print("Starting YOLO training...")
    print(f"Using device: {DEVICE}")

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Could not find data file: {DATA_YAML}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,

        # optimizer settings
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # augmentation
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        fliplr=0.5,
        degrees=3.0,
        translate=0.1,
        scale=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # other useful settings
        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        verbose=True
    )

    print("Training finished.")

    print("\nRunning validation on best weights...")
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    best_model = YOLO(best_model_path)
    metrics = best_model.val(data=DATA_YAML, imgsz=IMG_SIZE, device=DEVICE)

    print("Validation finished.")
    print(metrics)


if __name__ == "__main__":
    main()