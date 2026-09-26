try:
    import xir
    print("xir OK:", xir.__file__)
    print("xir attrs:", [a for a in dir(xir) if not a.startswith("_")][:20])
except Exception as e:
    print("xir FAILED:", e)
try:
    import vart
    print("vart OK")
except Exception as e:
    print("vart FAILED:", e)
