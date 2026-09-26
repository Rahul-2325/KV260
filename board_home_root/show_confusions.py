import cv2

for stem in ["WEB03809", "PublicDataset01011"]:
    img = cv2.imread("/home/root/valset/images/%s.jpg" % stem)
    lbl = open("/home/root/valset/labels/%s.txt" % stem).read()
    print(stem, img.shape, "labels:", lbl.strip().replace("\n", " | "))
    cv2.imwrite("/home/root/confusion_%s.jpg" % stem, img)
