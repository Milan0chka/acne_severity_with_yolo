import argparse
import yaml
from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path

def compute_features(model, img_path, conf_thres=0.2, iou_thres=0.7):
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

def get_model_name(run_dir: Path) -> str:
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        raise FileNotFoundError(f"Missing args.yaml in: {run_dir}")

    with open(args_yaml, "r") as f:
        meta = yaml.safe_load(f)

    model_value = meta.get("model", "model")
    return Path(str(model_value)).stem  # removes .pt


def main():
    parser = argparse.ArgumentParser(description="Extract YOLO features")

    parser.add_argument(
        "--run",
        type=str,
        default="train15",
        help="Run folder inside ../runs/detect/"
    )

    parser.add_argument(
        "--images",
        type=str,
        default="../data/raw",
        help="Directory with input images"
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Custom output name (without .csv). If omitted → auto features_{model}_{run}"
    )

    args = parser.parse_args()

    run_dir = Path(f"../runs/detect/{args.run}")
    model_name = get_model_name(run_dir)

    model_path = run_dir / "weights" / "best.pt"
    images_dir = Path(args.images)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_name = args.out if args.out else f"features_{model_name}_{args.run}"
    out_csv = Path(f"../data/features/{out_name}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRun: {args.run}")
    print(f"Model: {model_name}")
    print(f"Output CSV: {out_csv}\n")

    model = YOLO(model_path)

    rows = []
    image_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png"))
    )

    for img_path in image_files:
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