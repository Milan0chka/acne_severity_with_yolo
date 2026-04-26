import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path


class YOLOv8GradCAM:
    def __init__(self, yolo: YOLO, target_layer=None):
        self.yolo = yolo
        self.net = yolo.model
        self.net.eval()

        self.target_layer = target_layer or self.net.model[-10]

        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def preprocess(self, img_path, size=640):
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size))

        x = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(next(self.net.parameters()).device)
        x.requires_grad_(True)
        return x, rgb

    def forward_raw(self, x):
        out = self.net(x)

        if isinstance(out, (list, tuple)):
            pred = out[0]
        else:
            pred = out

        return pred

    def generate(self, img_path):
        x, original = self.preprocess(img_path)

        self.net.zero_grad(set_to_none=True)
        self.activations, self.gradients = None, None

        pred = self.forward_raw(x)

        if pred.dim() == 3 and pred.shape[1] < pred.shape[2]:
            pred = pred.permute(0, 2, 1)

        class_scores = pred[..., 4:]
        target = class_scores.max()

        target.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            return None, original

        acts = self.activations[0].detach().cpu().numpy()
        grads = self.gradients[0].detach().cpu().numpy()

        weights = grads.mean(axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()

        return cam, original

    def save_viz(self, img_path, out_path, alpha=0.35):
        cam, original = self.generate(img_path)

        if cam is None:
            # just save original
            cv2.imwrite(str(out_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
            return

        overlay = self.overlay_cam(original, cam, alpha=alpha)

        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


    def overlay_cam(self, original_rgb, cam, alpha=0.35):
        h, w = original_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=6)

        heatmap_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay



def main():
    import random

    model_path = "../runs/detect/train8/weights/best.pt"
    images_dir = Path("../data/raw")
    output_dir = Path("../results/gradcam")

    max_images = 200
    random_seed = 42
    conf_threshold = 0.25

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print("Initializing Grad-CAM...")
    gradcam = YOLOv8GradCAM(model)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png"))
    )

    if not image_files:
        raise RuntimeError(f"No images found in {images_dir.resolve()}")

    # Set seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    if max_images and len(image_files) > max_images:
        image_files = random.sample(image_files, k=max_images)

    print(f"\nProcessing {len(image_files)} random images...\n")

    success_count = 0
    no_detection_count = 0
    error_count = 0

    for i, img_path in enumerate(image_files, 1):
        save_path = output_dir / f"gradcam_{img_path.stem}.png"

        try:
            gradcam.save_viz(img_path, save_path)

            # check detection count
            results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
            n_detections = len(results[0].boxes) if results[0].boxes is not None else 0

            if n_detections > 0:
                print(f"[{i}/{len(image_files)}] ✓ {img_path.name} - {n_detections} detections")
                success_count += 1
            else:
                print(f"[{i}/{len(image_files)}] ○ {img_path.name} - No detections")
                no_detection_count += 1

        except Exception as e:
            print(f"[{i}/{len(image_files)}] ✗ {img_path.name} - Error: {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful Grad-CAMs: {success_count}")
    print(f"No detections: {no_detection_count}")
    print(f"Errors: {error_count}")
    print(f"\nVisualizations saved to: {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()