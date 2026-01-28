from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path

def compute_features(model, img_path, conf_thres=0.25, iou_thres=0.7):
    # Read image to get dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    img_area = w * h

    results = model.predict(source=str(img_path), conf=conf_thres, iou=iou_thres, verbose=False)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0.0, 0.0, 0  # density, avg_conf, count

    # coordinates of acne
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    total_acne_area = 0.0
    for x1, y1, x2, y2 in xyxy:
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        total_acne_area += bw * bh

    density = float(total_acne_area / img_area) if img_area > 0 else 0.0
    avg_conf = float(confs.mean()) if len(confs) > 0 else 0.0

    return density, avg_conf, int(len(confs))

def main():
    model_path = "../runs/detect/train4/weights/best.pt"
    model = YOLO(model_path)

    images_dir = Path("../data/raw")
    out_csv = Path("../data/features/features2.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))):
        density, avg_conf, n = compute_features(model, img_path)
        rows.append({
            "image": img_path.name,
            "density": density,
            "avg_conf": avg_conf,
            "n_detections": n
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    main()
