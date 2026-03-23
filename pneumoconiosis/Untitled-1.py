#!/usr/bin/env python3
"""
pneumo_inference.py

Accepts an input chest X-ray/CT image, runs lung segmentation (U-Net)
and stage classification (EfficientNet-B0), and outputs:
 - overlay image (input + segmentation mask)
 - classification probabilities and predicted label

Usage:
    python pneumo_inference.py --image path/to/xray.jpg \
        --seg-model path/to/seg_checkpoint.pth \
        --cls-model path/to/cls_checkpoint.pth \
        --outdir results/

If seg-model or cls-model are omitted, reasonable defaults (random or ImageNet
backbone) will be used but results won't be clinically valid until you supply
your trained checkpoints.
"""

import os
import argparse
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms.functional import to_pil_image

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
        diffX = x2.size()[3] - x2.size()[3]
        x = x[:, :, diffY // 2:diffY // 2 + x2.size()[2], diffX // 2:diffX // 2 + x2.size()[3]]
        x = torch.cat([x, x2], dim=1); x = self.conv1(x)
        # small skip with x1: upsample if needed
        # To keep shapes, we can do a final conv to map to base_c*2 -> out
        out = self.outc(x)
        return out

# -------------------------
# Classifier: EfficientNet-B0 (from torchvision)
# -------------------------
class PneumoClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Use torchvision's EfficientNet B0
        try:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = self.backbone.classifier[1].in_features
            # replace head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_f, num_classes)
            )
        except Exception as e:
            # Fallback: simple small CNN if torchvision version missing
            print("Warning: EfficientNet not available, using fallback small CNN. Error:", e)
            self.backback = nn.Sequential(
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
    """Load image as PIL. mode='L' for greyscale X-ray, 'RGB' for 3-channel."""
    img = Image.open(path).convert(mode)
    return img

def preprocess_for_seg(img_pil, target_size=256):
    """Return tensor for UNet: shape (1, H, W), normalized 0-1"""
    img = img_pil.copy()
    # optionally apply slight denoise
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.equalize(img)  # histogram equalization may help X-rays
    img = img.resize((target_size, target_size))
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)  # to gray
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor

def preprocess_for_cls(img_pil, target_size=224):
    """Return 3-channel tensor normalized for EfficientNet"""
    img = img_pil.copy().convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(img).unsqueeze(0)  # (1,3,H,W)

def postprocess_mask(mask_logits):
    """Sigmoid + threshold to binary mask. mask_logits: torch tensor (1,1,H,W)"""
    prob = torch.sigmoid(mask_logits)
    prob_np = prob.detach().cpu().squeeze().numpy()
    mask = (prob_np >= 0.5).astype(np.uint8)
    return mask, prob_np

def overlay_mask_on_image(img_pil, mask, alpha=0.4, mask_color=(255,0,0)):
    """Overlay binary mask (H,W) on RGB PIL image."""
    rgb = img_pil.convert('RGB').resize((mask.shape[1], mask.shape[0]))
    overlay = Image.new('RGBA', rgb.size)
    mask_img = Image.fromarray((mask*255).astype(np.uint8)).convert('L')
    color_layer = Image.new('RGBA', rgb.size, mask_color + (0,))
    # Put mask color where mask==1
    colored = Image.new('RGBA', rgb.size, mask_color + (0,))
    mask_alpha = mask_img.point(lambda p: int(p * alpha))
    colored.putalpha(mask_alpha)
    out = Image.alpha_composite(rgb.convert('RGBA'), colored)
    return out.convert('RGB')

# -------------------------
# Inference pipeline
# -------------------------
def run_inference(image_path, seg_model_path=None, cls_model_path=None, outdir='results',device='cpu', num_classes=3):
    os.makedirs(outdir, exist_ok=True)
    # Load models
    # segmentation
    seg_net = UNet(in_channels=1, out_channels=1, base_c=32)
    seg_net.to(device)
    if seg_model_path and os.path.exists(seg_model_path):
        ck = torch.load(seg_model_path, map_location=device)
        seg_net.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck)
        print("[INFO] Loaded segmentation weights:", seg_model_path)
    else:
        print("[WARN] No seg checkpoint provided or file missing. Using untrained UNet (results not reliable).")
    seg_net.eval()

    # classifier
    cls_net = PneumoClassifier(num_classes=num_classes, pretrained=True)
    cls_net.to(device)
    if cls_model_path and os.path.exists(cls_model_path):
        ck = torch.load(cls_model_path, map_location=device)
        cls_net.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck)
        print("[INFO] Loaded classifier weights:", cls_model_path)
    else:
        print("[WARN] No cls checkpoint provided or file missing. Using ImageNet-pretrained backbone (not fine-tuned).")

    # Load and preprocess image
    orig = load_image(image_path, mode='L')  # X-rays usually single-channel
    seg_in = preprocess_for_seg(orig, target_size=256).to(device)  # (1,1,256,256)
    cls_in = preprocess_for_cls(orig, target_size=224).to(device)   # (1,3,224,224)

    # segmentation forward
    with torch.no_grad():
        seg_logits = seg_net(seg_in)  # (1,1,H,W)
        mask_bin, mask_prob = postprocess_mask(seg_logits)

        # Optionally, apply mask to original and feed masked image to classifier
        # create masked RGB where background zeroed
        mask_resized_for_cls = Image.fromarray((mask_bin*255).astype(np.uint8)).resize((224,224))
        # convert to numpy boolean
        mask_bool = np.array(mask_resized_for_cls) > 0
        cls_np = cls_in.detach().cpu().squeeze().numpy()  # (3,H,W)
        # apply mask to each channel
        cls_np_masked = cls_np.copy()
        for c in range(cls_np_masked.shape[0]):
            cls_np_masked[c] = cls_np_masked[c] * mask_bool
        cls_in_masked = torch.from_numpy(cls_np_masked).unsqueeze(0).to(device).float()

        # classification (we'll run on masked input to focus on lung regions)
        logits = cls_net(cls_in_masked)
        probs = F.softmax(logits, dim=1).detach().cpu().squeeze().numpy()

    # prepare outputs
    classes = [f"Stage_{i+1}" for i in range(num_classes)]
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]

    # visualize overlay mask
    overlay = overlay_mask_on_image(orig.resize((256,256)), mask_bin, alpha=0.45)

    # Save overlay and a small report
    overlay_path = os.path.join(outdir, "overlay.png")
    overlay.save(overlay_path)

    # Save a textual report
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
# Optionally: simple training loop stubs (useful when you want to train)
# -------------------------
def train_segmentation(train_loader, val_loader, seg_net, epochs=10, device='cpu', outpath='seg_ckpt.pth'):
    optimizer = torch.optim.Adam(seg_net.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    seg_net.to(device)
    for ep in range(epochs):
        seg_net.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device); masks = masks.to(device)
            logits = seg_net(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
        print(f"Epoch {ep+1}/{epochs} train loss {running/len(train_loader):.4f}")
        # validation omitted for brevity
    torch.save({'model_state_dict': seg_net.state_dict()}, outpath)
    print("Saved seg checkpoint to", outpath)

def train_classifier(train_loader, val_loader, cls_net, epochs=10, device='cpu', outpath='cls_ckpt.pth'):
    optimizer = torch.optim.Adam(cls_net.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    cls_net.to(device)
    for ep in range(epochs):
        cls_net.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            logits = cls_net(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
        print(f"Epoch {ep+1}/{epochs} train loss {running/len(train_loader):.4f}")
    torch.save({'model_state_dict': cls_net.state_dict()}, outpath)
    print("Saved cls checkpoint to", outpath)

# -------------------------
# CLI (Modified for Colab usage)
# -------------------------
def main():
    # Create a dummy image for demonstration if one isn't provided
    dummy_image_path = "dummy_xray.png"
    # if not os.path.exists(dummy_image_path):
    #     print("[INFO] Creating a dummy image for demonstration.")
    #     img = Image.new('L', (256, 256), color = 'white')
    #     img.save(dummy_image_path)

    # Define the image path and other arguments directly
    image_path_to_use = "/content/person1_bacteria_1.jpeg" # Replace with your image path
    seg_model_path_to_use = None # Replace with path to your seg model or None
    cls_model_path_to_use = None # Replace with path to your cls model or None
    outdir_to_use = "results"
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes_to_use = 3

    res = run_inference(image_path_to_use, seg_model_path=seg_model_path_to_use,
                        cls_model_path=cls_model_path_to_use, outdir=outdir_to_use,
                        device=device_to_use, num_classes=num_classes_to_use)
    print("Done. Outputs:", res)

if __name__ == "__main__":
    main()