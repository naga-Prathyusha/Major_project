#!/usr/bin/env python3
"""
pneumo_inference.py

Inference pipeline: UNet segmentation + classifier
Saves overlay images and per-image report files.
"""
import os
import argparse
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# -------------------------
# U-Net
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xm = self.mid(x4)
        x = self.up3(xm)
        x = F.interpolate(x, size=(x4.size(2), x4.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x4], dim=1); x = self.conv3(x)
        x = self.up2(x)
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1); x = self.conv2(x)
        x = self.up1(x)
        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1); x = self.conv1(x)
        out = self.outc(x)
        return out

# -------------------------
# Classifier (EfficientNet B0 or fallback)
# -------------------------
class PneumoClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        try:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(in_f, num_classes))
        except Exception as e:
            print("Warning: EfficientNet not available, fallback used. Err:", e)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten(), nn.Linear(64, num_classes)
            )
    def forward(self, x): return self.backbone(x)

# -------------------------
# Utilities
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
    if arr.ndim == 3: arr = arr.mean(axis=2)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor

def preprocess_for_cls(img_pil, target_size=224):
    img = img_pil.copy().convert('RGB')
    transform = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
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
# run_inference (returns human-friendly messages)
# -------------------------
def run_inference(image_path, seg_model_path=None, cls_model_path=None, outdir='results', device='cpu', num_classes=3):
    os.makedirs(outdir, exist_ok=True)

    seg_net = UNet(in_channels=1, out_channels=1, base_c=32).to(device)
    if seg_model_path and os.path.exists(seg_model_path):
        ck = torch.load(seg_model_path, map_location=device); seg_net.load_state_dict(ck.get('model_state_dict', ck))
        print("[INFO] Loaded segmentation weights:", seg_model_path)
    else:
        print("[WARN] No seg checkpoint provided or file missing. Using untrained UNet (results not reliable).")
    seg_net.eval()

    cls_net = PneumoClassifier(num_classes=num_classes, pretrained=True).to(device)
    if cls_model_path and os.path.exists(cls_model_path):
        ck = torch.load(cls_model_path, map_location=device); cls_net.load_state_dict(ck.get('model_state_dict', ck))
        print("[INFO] Loaded classifier weights:", cls_model_path)
    else:
        print("[WARN] No cls checkpoint provided or file missing. Using ImageNet-pretrained backbone (not fine-tuned).")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
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

    # static mapping (change if you trained with different order)
    label_names = {0: "Normal", 1: "Bacteria infection", 2: "Virus (critical)"}
    human_messages = {0: "It is normal.", 1: "Bacteria has been found — please consult a doctor.", 2: "Critical condition — you must consult a doctor immediately."}

    pred_idx = int(np.argmax(probs))
    pred_label = label_names.get(pred_idx, f"Stage_{pred_idx+1}")
    human_message = human_messages.get(pred_idx, "")

    # save overlay and per-image report (unique filename)
    base = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(outdir, f"{base}_overlay.png")
    report_path = os.path.join(outdir, f"report_{base}.txt")
    overlay = overlay_mask_on_image(orig.resize((256,256)), mask_bin, alpha=0.45)
    overlay.save(overlay_path)

    with open(report_path, 'w') as f:
        f.write(f"Input image: {image_path}\n")
        f.write(f"Predicted label: {pred_label}\n")
        f.write(f"Message: {human_message}\n")
        f.write("Class probabilities:\n")
        for i,p in enumerate(probs):
            f.write(f"  {label_names.get(i,f'Stage_{i+1}')} : {p:.4f}\n")
        f.write("\nNotes:\n - Seg model: {}\n - Cls model: {}\n".format(seg_model_path or "None", cls_model_path or "None"))

    print("[INFO] Saved overlay to", overlay_path)
    print("[INFO] Saved report to", report_path)
    print("[RESULT] Prediction:", pred_label)
    print("         Message:", human_message)
    print("         Probabilities:", {label_names.get(i,f"Stage_{i+1}"): float(probs[i]) for i in range(len(probs))})
    return {"overlay": overlay_path, "report": report_path, "prediction": pred_label, "message": human_message, "probabilities": {label_names.get(i,f"Stage_{i+1}"): float(probs[i]) for i in range(len(probs))}}

# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--seg-model', default=None)
    parser.add_argument('--cls-model', default=None)
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--num-classes', type=int, default=3)
    args = parser.parse_args()
    run_inference(args.image, seg_model_path=args.seg_model, cls_model_path=args.cls_model, outdir=args.outdir, device=args.device, num_classes=args.num_classes)

if __name__ == "__main__": main()
