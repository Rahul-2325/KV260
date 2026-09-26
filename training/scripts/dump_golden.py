"""
dump_golden.py - Generate golden-reference arrays for the decode HLS IP.
Runs the same quantized model as decode_check.py, then saves the decode
stage's input (raw logits), constant tables (grid, stride), and outputs
(decoded boxes + class scores, in letterbox space, pre-NMS).
"""
import os, sys
sys.path.insert(0, '/workspace/YOLOX')
import torch, torch.nn as nn
import numpy as np, cv2
from pytorch_nndct.apis import torch_quantizer

WS = '/workspace'
CKPT = f'{WS}/models/lcam_yolox_wildfire_v32_compat.pth'
QDIR = f'{WS}/exports/detector_v2'
IMG  = f'{WS}/dataset/data/test/images/WEB09971.jpg'
OUTDIR = f'{WS}/golden'
SIZE = 640
STRIDES = [8, 16, 32]
os.makedirs(OUTDIR, exist_ok=True)

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
    grids, expanded = [], []
    for s in strides:
        hg = wg = size // s
        yv, xv = np.meshgrid(np.arange(hg), np.arange(wg), indexing='ij')
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), s))
    return np.concatenate(grids, 0), np.concatenate(expanded, 0)

def main():
    dev = torch.device('cpu')
    m = LCAMYOLOXDetector()
    m.model.load_state_dict(torch.load(CKPT, map_location='cpu')['model'], strict=False)
    m = m.to(dev); m.eval()
    q = torch_quantizer(quant_mode='test', module=m,
                        input_args=(torch.randn(1,3,SIZE,SIZE),),
                        output_dir=QDIR, device=dev)
    qm = q.quant_model; qm.eval()

    img = cv2.imread(IMG)
    x, ratio = preproc(img, SIZE)
    with torch.no_grad():
        out = qm(torch.from_numpy(x).unsqueeze(0))
    raw = (out[0] if isinstance(out,(list,tuple)) else out).detach().cpu().numpy()[0].astype(np.float32)  # (8400,7)

    grid, strd = make_grids(SIZE, STRIDES)
    grid = grid.astype(np.float32)              # (8400,2)
    strd = strd.astype(np.float32)              # (8400,1)

    # Decode (letterbox space) -- exactly matching decode_check.py
    cx = (raw[:,0] + grid[:,0]) * strd[:,0]
    cy = (raw[:,1] + grid[:,1]) * strd[:,0]
    ww = np.exp(raw[:,2]) * strd[:,0]
    hh = np.exp(raw[:,3]) * strd[:,0]
    sig = lambda z: 1/(1+np.exp(-z))
    obj = sig(raw[:,4]); sm = sig(raw[:,5]); fr = sig(raw[:,6])
    cs = obj*sm; cf = obj*fr

    # decode output: [cx, cy, w, h, score_smoke, score_fire]  (8400,6)
    decoded = np.stack([cx, cy, ww, hh, cs, cf], axis=1).astype(np.float32)

    # intermediates for stage-by-stage HLS verification
    exp_w = np.exp(raw[:,2]).astype(np.float32)
    exp_h = np.exp(raw[:,3]).astype(np.float32)
    sig_obj = obj.astype(np.float32)
    sig_sm  = sm.astype(np.float32)
    sig_fr  = fr.astype(np.float32)

    np.save(f'{OUTDIR}/decode_input.npy', raw)        # (8400,7)
    np.save(f'{OUTDIR}/grid.npy', grid)               # (8400,2)
    np.save(f'{OUTDIR}/stride.npy', strd)             # (8400,1)
    np.save(f'{OUTDIR}/decode_output.npy', decoded)   # (8400,6)
    np.savez(f'{OUTDIR}/intermediates.npz',
             exp_w=exp_w, exp_h=exp_h,
             sig_obj=sig_obj, sig_sm=sig_sm, sig_fr=sig_fr)

    # quick stats so we can eyeball ranges (critical for fixed-point design)
    print("raw    range:", raw.min(), raw.max())
    print("raw[:, 2:4] (w,h logits) range:", raw[:,2:4].min(), raw[:,2:4].max(), "  <- exp() input range")
    print("exp_w  range:", exp_w.min(), exp_w.max())
    print("decoded cx/cy range:", decoded[:,0:2].min(), decoded[:,0:2].max())
    print("decoded w/h  range:", decoded[:,2:4].min(), decoded[:,2:4].max())
    print("scores range:", decoded[:,4:6].min(), decoded[:,4:6].max())
    print("\nSaved golden arrays to", OUTDIR)
    for f in ['decode_input.npy','grid.npy','stride.npy','decode_output.npy','intermediates.npz']:
        print("  ", f)

main()
