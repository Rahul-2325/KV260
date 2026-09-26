"""
decode_check.py - Decode quantized detector_v2 output into real boxes,
run per-class NMS, compare to WEB09971 ground truth, draw and save.
Run inside Docker: python /workspace/scripts/decode_check.py
"""
import os, sys
sys.path.insert(0, '/workspace/YOLOX')
import torch, torch.nn as nn
import numpy as np, cv2
from pytorch_nndct.apis import torch_quantizer

WS = '/workspace'
CKPT = f'{WS}/models/lcam_yolox_v5_compat.pth'
QDIR = f'{WS}/exports/detector_v5'
IMG  = f'{WS}/dataset/data/test/images/WEB09971.jpg'
OUT  = f'{WS}/decode_check_v5_WEB09971.jpg'
SIZE = 640
STRIDES = [8, 16, 32]
CONF = 0.40
NMS_IOU = 0.45
CLS = ['smoke', 'fire']
# Ground truth (YOLO normalized): smoke center, fire lower-center
GT = [(0, 0.4921, 0.4092, 0.3292, 0.8150),
      (1, 0.4900, 0.7542, 0.1767, 0.1183)]


class LCAMYOLOXDetector(nn.Module):
    def __init__(self):
        super().__init__()
        from yolox.exp import get_exp
        exp = get_exp(exp_file=None, exp_name='yolox-s')
        exp.num_classes = 2; exp.act = 'lrelu'
        self.model = exp.get_model(); self.model.eval()
        self.model.head.decode_in_inference = False
    def forward(self, x): return self.model(x)


def preproc(img, size=640):
    h0, w0 = img.shape[:2]
    r = min(size/h0, size/w0)
    nw, nh = int(w0*r), int(h0*r)
    res = cv2.resize(img, (nw, nh))
    pad = np.full((size, size, 3), 114, np.uint8)
    pad[:nh, :nw] = res
    return pad.astype(np.float32).transpose(2,0,1), r


def make_grids(size=640, strides=(8,16,32)):
    """Standard YOLOX grid+stride for the 8400 predictions."""
    grids, expanded = [], []
    for s in strides:
        hg = wg = size // s
        yv, xv = np.meshgrid(np.arange(hg), np.arange(wg), indexing='ij')
        grid = np.stack((xv, yv), 2).reshape(-1, 2)   # (hg*wg, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), s))
    return np.concatenate(grids, 0), np.concatenate(expanded, 0)  # (8400,2),(8400,1)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = max(0, x2-x1); h = max(0, y2-y1)
    inter = w*h
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0


def nms(dets):  # dets: list of [x1,y1,x2,y2,score,cls]
    dets = sorted(dets, key=lambda d: -d[4])
    keep = []
    while dets:
        best = dets.pop(0); keep.append(best)
        dets = [d for d in dets if d[5] != best[5] or iou(best, d) < NMS_IOU]
    return keep


def main():
    dev = torch.device('cpu')
    m = LCAMYOLOXDetector()
    m.model.load_state_dict((lambda c: c['model'] if isinstance(c,dict) and 'model' in c else c)(torch.load(CKPT, map_location='cpu')), strict=False)
    m = m.to(dev); m.eval()
    q = torch_quantizer(quant_mode='test', module=m,
                        input_args=(torch.randn(1,3,SIZE,SIZE),),
                        output_dir=QDIR, device=dev)
    qm = q.quant_model; qm.eval()

    img = cv2.imread(IMG); H, W = img.shape[:2]
    x, ratio = preproc(img, SIZE)
    with torch.no_grad():
        out = qm(torch.from_numpy(x).unsqueeze(0))
    raw = (out[0] if isinstance(out,(list,tuple)) else out).detach().cpu().numpy()[0]  # (8400,7)

    grid, strd = make_grids(SIZE, STRIDES)
    # decode boxes (letterbox space)
    cx = (raw[:,0] + grid[:,0]) * strd[:,0]
    cy = (raw[:,1] + grid[:,1]) * strd[:,0]
    ww = np.exp(raw[:,2]) * strd[:,0]
    hh = np.exp(raw[:,3]) * strd[:,0]
    sig = lambda z: 1/(1+np.exp(-z))
    obj = sig(raw[:,4]); sm = sig(raw[:,5]); fr = sig(raw[:,6])
    cs = obj*sm; cf = obj*fr

    dets = []
    for i in range(raw.shape[0]):
        for c, score in ((0, cs[i]), (1, cf[i])):
            if score > CONF:
                # to corner, undo letterbox ratio -> original pixels
                x1 = (cx[i]-ww[i]/2)/ratio; y1 = (cy[i]-hh[i]/2)/ratio
                x2 = (cx[i]+ww[i]/2)/ratio; y2 = (cy[i]+hh[i]/2)/ratio
                dets.append([x1, y1, x2, y2, float(score), c])
    print(f"Pre-NMS detections > {CONF}: {len(dets)}")
    kept = nms(dets)
    print(f"Post-NMS detections: {len(kept)}\n")

    for d in sorted(kept, key=lambda d:-d[4]):
        print(f"  {CLS[d[5]]:5s} score={d[4]:.3f} box=({d[0]:.0f},{d[1]:.0f})-({d[2]:.0f},{d[3]:.0f})")

    # Ground truth in pixels
    print("\n--- GROUND TRUTH (pixels) ---")
    gt_px = []
    for c, gx, gy, gw, gh in GT:
        bx1=(gx-gw/2)*W; by1=(gy-gh/2)*H; bx2=(gx+gw/2)*W; by2=(gy+gh/2)*H
        gt_px.append((c,bx1,by1,bx2,by2))
        print(f"  {CLS[c]:5s} box=({bx1:.0f},{by1:.0f})-({bx2:.0f},{by2:.0f})")

    # Match: best IoU between each GT and our same-class detections
    print("\n--- MATCH (IoU of best same-class detection vs GT) ---")
    for c,bx1,by1,bx2,by2 in gt_px:
        best=0
        for d in kept:
            if d[5]==c:
                best=max(best, iou([bx1,by1,bx2,by2], d[:4]))
        verdict = "GOOD" if best>0.3 else ("WEAK" if best>0.1 else "MISS")
        print(f"  {CLS[c]:5s}: best IoU={best:.3f}  [{verdict}]")

    # Draw
    vis = img.copy()
    for c,bx1,by1,bx2,by2 in gt_px:   # GT in green
        cv2.rectangle(vis,(int(bx1),int(by1)),(int(bx2),int(by2)),(0,255,0),2)
        cv2.putText(vis,f"GT-{CLS[c]}",(int(bx1),int(by1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    for d in kept:                     # detections in red
        cv2.rectangle(vis,(int(d[0]),int(d[1])),(int(d[2]),int(d[3])),(0,0,255),2)
        cv2.putText(vis,f"{CLS[d[5]]} {d[4]:.2f}",(int(d[0]),int(d[1])-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.imwrite(OUT, vis)
    print(f"\nSaved annotated image: {OUT}")
    print("  GREEN = ground truth, RED = detections")


main()
