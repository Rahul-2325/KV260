import cv2
import glob

fs = sorted(glob.glob('/home/root/valset/images/*'))
print("total images:", len(fs))
shapes = {}
for f in fs:
    img = cv2.imread(f)
    if img is None:
        continue
    shapes[img.shape] = shapes.get(img.shape, 0) + 1
for shp, n in sorted(shapes.items(), key=lambda kv: -kv[1]):
    print("  shape", shp, "x", n)
