"""
Select fire-prominent images for calibration using HSV color heuristics.
Fire = bright, saturated orange/red/yellow pixels. No labels needed.
"""
import os, cv2, numpy as np

CALIB_DIR = os.path.expanduser('~/wildfire_project/calibration')
OUT_LIST  = os.path.expanduser('~/wildfire_project/calibration_fire_list.txt')

def fire_score(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    hue_fire = ((h <= 35) | (h >= 160))      # red-orange-yellow
    bright   = (v > 150) & (s > 90)          # bright + saturated
    return float((hue_fire & bright).mean())

files = sorted([f for f in os.listdir(CALIB_DIR)
                if f.lower().endswith(('.jpg','.jpeg','.png'))])
print("Scoring %d images..." % len(files))
scored = []
for i, f in enumerate(files):
    img = cv2.imread(os.path.join(CALIB_DIR, f))
    if img is None: continue
    scored.append((fire_score(img), f))
    if (i+1) % 100 == 0: print("  %d/%d" % (i+1, len(files)))

scored.sort(reverse=True)
print("\nTop 10 most fire-colored:")
for sc, f in scored[:10]:
    print("  %.4f  %s" % (sc, f))
print("\nMedian score:", round(np.median([s for s,_ in scored]),4))

fire_rich = [f for _, f in scored[:180]]
rest = [f for _, f in scored[180:]]
varied_idx = np.linspace(0, len(rest)-1, 120).astype(int) if rest else []
varied = [rest[i] for i in varied_idx]
final = fire_rich + varied

with open(OUT_LIST, 'w') as fh:
    for f in final:
        fh.write(f + "\n")
print("\nWrote %d images (180 fire-rich + %d varied) to %s" % (len(final), len(varied), OUT_LIST))
import re
pref = {}
for f in final:
    p = re.match(r'[A-Za-z]+', f); p = p.group() if p else '?'
    pref[p] = pref.get(p,0)+1
print("Prefix mix:", pref)
