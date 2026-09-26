#!/usr/bin/env python3
"""
LCAM-YOLOX Board Inference Script
For KV260 with Vitis-AI 3.0 runtime

Usage:
    python3 infer_kv260.py --xclbin /path/to/kv260_lcam.xclbin \
                           --xmodel /path/to/lcam_v5.xmodel \
                           --image  /path/to/test.jpg

Requirements (on KV260 board):
    pip3 install opencv-python-headless numpy
    Vitis-AI runtime must be installed (vart, xir)
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# ─── Constants ────────────────────────────────────────────────────────────────
INPUT_W     = 640
INPUT_H     = 640
NUM_CLASSES = 2          # fire / smoke (adjust if your model has different classes)
CONF_THRESH = 0.3
NMS_THRESH  = 0.45
CLASS_NAMES = ["fire", "smoke"]   # adjust to match your training labels

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(image_path):
    """Load image, letterbox-resize to 640x640, normalize to [0,1]."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img.shape[:2]

    # Letterbox resize
    scale = min(INPUT_W / orig_w, INPUT_H / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to 640x640
    canvas = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
    pad_x = (INPUT_W - new_w) // 2
    pad_y = (INPUT_H - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    # Normalize [0, 255] → [0, 1], then to float32
    inp = canvas.astype(np.float32) / 255.0

    # DPU expects NHWC layout
    inp = inp[np.newaxis, ...]   # (1, 640, 640, 3)

    return inp, orig_h, orig_w, scale, pad_x, pad_y


# ─── Postprocessing ────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(outputs, orig_h, orig_w, scale, pad_x, pad_y,
                conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
    """
    Decode YOLOX head outputs and apply NMS.
    outputs: list of numpy arrays from DPU runner
    Expected: one tensor of shape (1, 8400, 7) or similar
    """
    # Concatenate all outputs if multiple tensors
    if len(outputs) == 1:
        pred = outputs[0].squeeze()   # (8400, 7) or similar
    else:
        pred = np.concatenate([o.squeeze() for o in outputs], axis=0)

    # YOLOX decode: [cx, cy, w, h, obj_conf, cls0, cls1, ...]
    obj_conf  = sigmoid(pred[:, 4:5])
    cls_conf  = sigmoid(pred[:, 5:])
    scores    = obj_conf * cls_conf          # (N, num_classes)

    # Filter by confidence
    max_scores = scores.max(axis=1)
    class_ids  = scores.argmax(axis=1)
    mask       = max_scores > conf_thresh

    if mask.sum() == 0:
        return [], [], []

    boxes_raw  = pred[mask, :4]
    max_scores = max_scores[mask]
    class_ids  = class_ids[mask]

    # cx, cy, w, h → x1, y1, x2, y2 (in input image space)
    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Remove letterbox padding and rescale to original image
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS per class
    keep_boxes, keep_scores, keep_cls = [], [], []
    for cls_id in np.unique(class_ids):
        idx = class_ids == cls_id
        b   = boxes[idx]
        s   = max_scores[idx]
        indices = cv2.dnn.NMSBoxes(
            b.tolist(), s.tolist(), conf_thresh, nms_thresh
        )
        if len(indices) > 0:
            for i in indices.flatten():
                keep_boxes.append(b[i])
                keep_scores.append(s[i])
                keep_cls.append(int(cls_id))

    return keep_boxes, keep_scores, keep_cls


# ─── Draw results ─────────────────────────────────────────────────────────────
def draw_results(image_path, boxes, scores, class_ids, output_path="result.jpg"):
    img = cv2.imread(image_path)
    colors = [(0, 0, 255), (0, 165, 255)]   # red=fire, orange=smoke

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")
    return img


# ─── Main inference ───────────────────────────────────────────────────────────
def run_inference(xclbin_path, xmodel_path, image_path, output_path="result.jpg"):
    # Import VART/XIR here so script gives clean error if not on board
    try:
        import xir
        import vart
    except ImportError:
        print("ERROR: xir/vart not found. Run this script on the KV260 board.")
        sys.exit(1)

    print(f"Loading xclbin: {xclbin_path}")
    print(f"Loading xmodel: {xmodel_path}")

    # ── Step 1: Load xmodel graph ──────────────────────────────────────────────
    g = xir.Graph.deserialize(xmodel_path)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()

    # Find DPU subgraphs
    dpu_subgraphs = [sg for sg in subgraphs
                     if sg.has_attr("device") and sg.get_attr("device") == "DPU"]

    if not dpu_subgraphs:
        print("ERROR: No DPU subgraphs found in xmodel")
        sys.exit(1)

    print(f"Found {len(dpu_subgraphs)} DPU subgraph(s)")

    # ── Step 2: Create DPU runner ──────────────────────────────────────────────
    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")

    # ── Step 3: Get input/output tensor info ───────────────────────────────────
    input_tensors  = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    print(f"Input  tensors: {[list(t.dims) for t in input_tensors]}")
    print(f"Output tensors: {[list(t.dims) for t in output_tensors]}")

    # ── Step 4: Preprocess image ───────────────────────────────────────────────
    print(f"Processing image: {image_path}")
    inp, orig_h, orig_w, scale, pad_x, pad_y = preprocess(image_path)

    # Allocate input/output buffers
    input_data  = [np.zeros(list(t.dims), dtype=np.float32)
                   for t in input_tensors]
    output_data = [np.zeros(list(t.dims), dtype=np.float32)
                   for t in output_tensors]

    # Copy preprocessed image into input buffer
    input_data[0][...] = inp

    # ── Step 5: Run inference ──────────────────────────────────────────────────
    print("Running DPU inference...")
    t0 = time.perf_counter()

    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Inference time: {elapsed:.2f} ms  ({1000/elapsed:.1f} FPS)")

    # ── Step 6: Postprocess ────────────────────────────────────────────────────
    boxes, scores, class_ids = postprocess(
        output_data, orig_h, orig_w, scale, pad_x, pad_y
    )

    print(f"Detections: {len(boxes)}")
    for box, score, cls_id in zip(boxes, scores, class_ids):
        print(f"  {CLASS_NAMES[cls_id]:8s} conf={score:.3f}  "
              f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    # ── Step 7: Draw and save ──────────────────────────────────────────────────
    draw_results(image_path, boxes, scores, class_ids, output_path)

    return boxes, scores, class_ids


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCAM-YOLOX inference on KV260")
    parser.add_argument("--xclbin", required=True,
                        help="Path to xclbin file")
    parser.add_argument("--xmodel", required=True,
                        help="Path to compiled xmodel file")
    parser.add_argument("--image",  required=True,
                        help="Path to input image (jpg/png)")
    parser.add_argument("--output", default="result.jpg",
                        help="Output image path (default: result.jpg)")
    parser.add_argument("--conf",   type=float, default=CONF_THRESH,
                        help=f"Confidence threshold (default: {CONF_THRESH})")
    args = parser.parse_args()

    # Set XRT device to use our xclbin
    os.environ["XLNX_VART_FIRMWARE"] = args.xclbin

    run_inference(args.xclbin, args.xmodel, args.image, args.output)
