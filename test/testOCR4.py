# debug_boxes_v2.py
# Same as before, but with: dual-polarity binarization (handles inverted tiles),
# slightly looser gates so the right column isn't clipped.

import cv2, numpy as np
from pathlib import Path

# --------------------------- KNOBS ---------------------------
IMG_PATH = Path("scorebug_test.png")

UPSCALE = 2
EQ_HIST = True

ADAPT_BLOCK = 31
ADAPT_C = 10

MORPH_CLOSE_W, MORPH_CLOSE_H = 5, 3

# ignore a small strip on the left (kills the flag)
EXCLUDE_LEFT_FRAC = 0.14

# keep right-edge tiebreak digits
DROP_BORDER_TOUCHING = False

# aspect and size guards
MIN_ASPECT_W_OVER_H = 0.30    # relaxed for squarish tiles
MAX_ASPECT_W_OVER_H = 18.0

THIN_H_FRAC = 0.12

MIN_HEIGHT_FRAC = 0.25
MAX_HEIGHT_FRAC = 1.30         # allow taller right column tiles

MIN_FILL = 0.05                # relaxed so hollow-ish digits pass
MAX_FILL = 0.88

MIN_AREA = 14                  # slight relax
MAX_AREA_FRAC = 0.40
# -------------------------------------------------------------

def imread_gray(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    if UPSCALE > 1:
        h, w = im.shape[:2]
        im = cv2.resize(im, (w*UPSCALE, h*UPSCALE), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def binarize(g):
    if EQ_HIST:
        g = cv2.equalizeHist(g)

    # Dual-polarity: capture dark-on-light and light-on-dark glyphs
    bw_a = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C
    )
    bw_b = cv2.adaptiveThreshold(
        255 - g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C
    )
    bw = cv2.bitwise_or(bw_a, bw_b)

    if MORPH_CLOSE_W and MORPH_CLOSE_H:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE_W, MORPH_CLOSE_H))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    return bw

def cc_all(bw):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    H, W = bw.shape
    items = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        items.append({
            "i": i,
            "bbox": (x, y, w, h),
            "area": int(area),
            "cx": float(cx), "cy": float(cy),
            "touch": (x <= 0 or y <= 0 or x+w >= W-1 or y+h >= H-1)
        })
    return items

def filter_text_like(items, img_size):
    W, H = img_size
    approx_row_h = H / 2.2
    left_cut = int(EXCLUDE_LEFT_FRAC * W)

    kept = []
    for it in items:
        x, y, w, h = it["bbox"]; area = it["area"]

        if x + w <= left_cut: continue
        if area < MIN_AREA: continue
        if area > MAX_AREA_FRAC * (W*H): continue
        if DROP_BORDER_TOUCHING and it["touch"]: continue

        if not (MIN_HEIGHT_FRAC * approx_row_h <= h <= MAX_HEIGHT_FRAC * approx_row_h):
            continue

        aspect = w / float(max(1, h))
        if aspect < MIN_ASPECT_W_OVER_H:  # too skinny
            continue
        if aspect > MAX_ASPECT_W_OVER_H:  # too flat
            continue

        fill = area / float(max(1, w*h))
        if not (MIN_FILL <= fill <= MAX_FILL):
            continue

        kept.append(it)
    return kept

def draw_boxes(gray, items, color, label, out_path):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for it in items:
        x, y, w, h = it["bbox"]
        cv2.rectangle(rgb, (x, y), (x+w, y+h), color, 2)
    cv2.putText(rgb, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, rgb)

def main():
    g = imread_gray(IMG_PATH)
    bw = binarize(g)
    cv2.imwrite("dbg_bin.png", bw)

    items = cc_all(bw)
    draw_boxes(g, items, (0,160,0), "ALL", "dbg_all_boxes.png")

    kept = filter_text_like(items, (g.shape[1], g.shape[0]))
    draw_boxes(g, kept, (0,255,255), "KEPT", "dbg_kept_boxes.png")

    print(f"components: total={len(items)} kept={len(kept)}")

if __name__ == "__main__":
    main()
