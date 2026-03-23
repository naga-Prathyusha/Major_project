# batch_infer.py
import os, glob, csv
from pneumo_inference import run_inference
OUTROOT = "batch_results"
os.makedirs(OUTROOT, exist_ok=True)
impaths = sorted([p for p in glob.glob(os.path.join("dataset_cls","**","*.*"), recursive=True) if p.lower().endswith(('.jpg','.jpeg','.png'))])
csv_path = os.path.join(OUTROOT, "batch_report.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image','prediction','message','overlay','report'])
    for img in impaths:
        try:
            res = run_inference(img, seg_model_path='seg_ckpt.pth', cls_model_path='cls_ckpt.pth', outdir=OUTROOT, device='cpu', num_classes=3)
            w.writerow([img, res['prediction'], res['message'], res['overlay'], res['report']])
        except Exception as e:
            w.writerow([img, 'ERROR', str(e), '', ''])
print("Batch done. Saved:", csv_path)
