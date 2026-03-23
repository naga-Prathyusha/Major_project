#!/usr/bin/env python3
"""
pneumo_inference.py

Usage:
    python pneumo_inference.py --image path/to/xray.jpg \
        [--seg-model path/to/seg_checkpoint.pth] \
        [--cls-model path/to/cls_checkpoint.pth] \
        [--outdir results/] [--device cpu] [--num-classes 3]
"""

import os
import argparse
fr
om PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# -------------------------
# U-Net (simple implementation)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_c=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c, base_c*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c*2, base_c*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c*4, base_c*8))
        self.mid = DoubleConv(base_c*8, base_c*16)
        self.up3 = nn.ConvTranspose2d(base_c*16, base_c*8, 2, stride=2)
        self.conv3 = DoubleConv(base_c*16, base_c*8)
        self.up2 = nn.ConvTranspose2d(base_c*8, base_c*4, 2, stride=2)
        self.conv2 = DoubleConv(base_c*8, base_c*4)
        self.up1 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, stride=2)
        self.conv1 = DoubleConv(base_c*4, base_c*2)
        self.outc = nn.Conv2d(base_c*2, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xm = self.mid(x4)
        # decoder
        x = self.up3(xm)
        # Crop x to match the spatial dimensions of x4
        diffY = x.size()[2] - x4.size()[2]
        diffX = x.size()[3] - x4.size()[3]
        x = x[:, :, diffY // 2:diffY // 2 + x4.size()[2], diffX // 2:diffX // 2 + x4.size()[3]]
        x = torch.cat([x, x4], dim=1); x = self.conv3(x)
        x = self.up2(x)
        # Crop x to match the spatial dimensions of x3
        diffY = x.size()[2] - x3.size()[2]
        diffX = x.size()[3] - x3.size()[3]
        x = x[:, :, diffY // 2:diffY // 2 + x3.size()[2], diffX // 2:diffX // 2 + x3.size()[3]]
        x = torch.cat([x, x3], dim=1); x = self.conv2(x)
        x = self.up1(x)
        # Crop x to match the spatial dimensions of x2
        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]   # <-- fixed typo here
        x = x[:, :, diffY // 2:diffY // 2 + x2.size()[2], diffX // 2:diffX // 2 + x2.size()[3]]
        x = torch.cat([x, x2], dim=1); x = self.conv1(x)
        out = self.outc(x)
        return out

# -------------------------
# Classifier: EfficientNet-B0 (from torchvision)
# -------------------------
class PneumoClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        try:
            # uses torchvision >=0.13 style weights enum
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_f, num_classes)
            )
        except Exception as e:
            # Fallback small CNN if EfficientNet unavailable locally
            print("Warning: EfficientNet not available, using fallback small CNN. Error:", e)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten(), nn.Linear(64, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)

# -------------------------
# Utility functions
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(path, mode='L'):
    img = Image.open(path).convert(mode)
    return img

def preprocess_for_seg(img_pil, target_size=256):
    img = img_pil.copy()
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.equalize(img)
    img = img.resize((target_size, target_size))
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor

def preprocess_for_cls(img_pil, target_size=224):
    img = img_pil.copy().convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(img).unsqueeze(0)

def postprocess_mask(mask_logits):
    prob = torch.sigmoid(mask_logits)
    prob_np = prob.detach().cpu().squeeze().numpy()
    mask = (prob_np >= 0.5).astype(np.uint8)
    return mask, prob_np

def overlay_mask_on_image(img_pil, mask, alpha=0.4, mask_color=(255,0,0)):
    rgb = img_pil.convert('RGB').resize((mask.shape[1], mask.shape[0]))
    mask_img = Image.fromarray((mask*255).astype(np.uint8)).convert('L')
    colored = Image.new('RGBA', rgb.size, mask_color + (0,))
    mask_alpha = mask_img.point(lambda p: int(p * alpha))
    colored.putalpha(mask_alpha)
    out = Image.alpha_composite(rgb.convert('RGBA'), colored)
    return out.convert('RGB')

# -------------------------
# Inference pipeline
# -------------------------
def _safe_load_state_dict(model, ck_path, device):
    ck = torch.load(ck_path, map_location=device)
    # ck might be a plain state_dict or a dict containing 'model_state_dict'
    if isinstance(ck, dict) and ('model_state_dict' in ck or 'state_dict' in ck):
        sd = ck.get('model_state_dict', ck.get('state_dict', ck))
    else:
        sd = ck
    model.load_state_dict(sd)
    return True

def run_inference(image_path, seg_model_path=None, cls_model_path=None, outdir='results',device='cpu', num_classes=3):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(device)
    seg_net = UNet(in_channels=1, out_channels=1, base_c=32).to(device)
    if seg_model_path and os.path.exists(seg_model_path):
        try:
            _safe_load_state_dict(seg_net, seg_model_path, device)
            print("[INFO] Loaded segmentation weights:", seg_model_path)
        except Exception as e:
            print("[WARN] Failed to load seg checkpoint:", e)
    else:
        print("[WARN] No seg checkpoint provided or file missing. Using untrained UNet (results not reliable).")
    seg_net.eval()

    cls_net = PneumoClassifier(num_classes=num_classes, pretrained=True).to(device)
    if cls_model_path and os.path.exists(cls_model_path):
        try:
            _safe_load_state_dict(cls_net, cls_model_path, device)
            print("[INFO] Loaded classifier weights:", cls_model_path)
        except Exception as e:
            print("[WARN] Failed to load cls checkpoint:", e)
    else:
        print("[WARN] No cls checkpoint provided or file missing. Using ImageNet-pretrained backbone (not fine-tuned).")

    orig = load_image(image_path, mode='L')
    seg_in = preprocess_for_seg(orig, target_size=256).to(device)
    cls_in = preprocess_for_cls(orig, target_size=224).to(device)

    with torch.no_grad():
        seg_logits = seg_net(seg_in)
        mask_bin, mask_prob = postprocess_mask(seg_logits)

        mask_resized_for_cls = Image.fromarray((mask_bin*255).astype(np.uint8)).resize((224,224))
        mask_bool = np.array(mask_resized_for_cls) > 0
        cls_np = cls_in.detach().cpu().squeeze().numpy()
        cls_np_masked = cls_np.copy()
        for c in range(cls_np_masked.shape[0]):
            cls_np_masked[c] = cls_np_masked[c] * mask_bool
        cls_in_masked = torch.from_numpy(cls_np_masked).unsqueeze(0).to(device).float()

        logits = cls_net(cls_in_masked)
        probs = F.softmax(logits, dim=1).detach().cpu().squeeze().numpy()

    classes = [f"Stage_{i+1}" for i in range(num_classes)]
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    overlay = overlay_mask_on_image(orig.resize((256,256)), mask_bin, alpha=0.45)
    overlay_path = os.path.join(outdir, "overlay.png")
    overlay.save(overlay_path)

    report_path = os.path.join(outdir, "report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Input image: {image_path}\n")
        f.write(f"Predicted label: {pred_label}\n")
        f.write("Class probabilities:\n")
        for i, p in enumerate(probs):
            f.write(f"  {classes[i]}: {p:.4f}\n")
        f.write("\nNotes:\n - Segmentation model: {}\n - Classifier model: {}\n".format(seg_model_path or "None", cls_model_path or "None"))

    print("[INFO] Saved overlay to", overlay_path)
    print("[INFO] Saved report to", report_path)
    print("[RESULT] Prediction:", pred_label)
    print("         Probabilities:", {classes[i]: float(probs[i]) for i in range(len(classes))})
    return {
        "overlay": overlay_path,
        "report": report_path,
        "prediction": pred_label,
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }

# -------------------------
# CLI entrypoint
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--seg-model", default=None, help="Path to segmentation checkpoint (optional)")
    p.add_argument("--cls-model", default=None, help="Path to classifier checkpoint (optional)")
    p.add_argument("--outdir", default="results", help="Output directory")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda (or torch device str)")
    p.add_argument("--num-classes", type=int, default=3, help="Number of classifier classes")
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device
    # allow "auto" detection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_inference(args.image, seg_model_path=args.seg_model,
                        cls_model_path=args.cls_model, outdir=args.outdir,
                        device=device, num_classes=args.num_classes)
    print("Done. Outputs:", res)

if __name__ == "__main__":
    main()
