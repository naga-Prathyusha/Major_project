# dataset_and_train.py
import os
from glob import glob
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = {os.path.splitext(os.path.basename(p))[0]:p for p in glob(os.path.join(masks_dir,'*'))}
        self.img_size = img_size
        self.to_tensor = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert('L')
        mask_path = self.masks.get(name, None)
        if mask_path is None:
            mask = Image.new('L', img.size, 0)
        else:
            mask = Image.open(mask_path).convert('L')
        img = self.to_tensor(img)
        mask = self.to_tensor(mask)
        mask = (mask >= 0.5).float()
        return img, mask, name

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.samples=[]
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            for p in glob(os.path.join(root_dir,c,'*')):
                self.samples.append((p, self.class_to_idx[c]))
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, os.path.basename(path)

def make_seg_loaders(images_dir, masks_dir, batch_size=4, img_size=256, split=0.8):
    ds = SegmentationDataset(images_dir, masks_dir, img_size=img_size)
    n=len(ds); idx=list(range(n)); random.shuffle(idx)
    cut=int(n*split)
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(ds, idx[:cut]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, idx[cut:]), batch_size=batch_size, shuffle=False) if cut < n else None
    return train_loader, val_loader

def make_cls_loaders(root_dir, batch_size=8, img_size=224, split=0.8):
    ds = ClassificationDataset(root_dir, img_size=img_size)
    n=len(ds); idx=list(range(n)); random.shuffle(idx)
    cut=int(n*split)
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(ds, idx[:cut]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, idx[cut:]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, ds.class_to_idx

def train_segmentation(seg_net, train_loader, val_loader=None, epochs=5, device='cpu', outpath='seg_ckpt.pth'):
    seg_net.to(device)
    opt = torch.optim.Adam(seg_net.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for ep in range(1, epochs+1):
        seg_net.train()
        total=0.0
        for imgs,masks,_ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = seg_net(imgs)
            loss = criterion(logits, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[SEG] Epoch {ep}/{epochs} train loss {total/len(train_loader):.4f}")
    torch.save({'model_state_dict': seg_net.state_dict()}, outpath)
    print("[SEG] Saved", outpath)

def train_classifier(cls_net, train_loader, val_loader=None, epochs=5, device='cpu', outpath='cls_ckpt.pth'):
    cls_net.to(device)
    opt = torch.optim.Adam(cls_net.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        cls_net.train()
        total=0.0
        for imgs,labels,_ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = cls_net(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[CLS] Epoch {ep}/{epochs} train loss {total/len(train_loader):.4f}")
    torch.save({'model_state_dict': cls_net.state_dict()}, outpath)
    print("[CLS] Saved", outpath)
