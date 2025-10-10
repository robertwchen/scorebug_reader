# debug_boxes_edges.py
# Polarity-agnostic detection of text-like boxes via Scharr edges + external contours.

import cv2, numpy as np
from pathlib import Path

# --------------------------- KNOBS ---------------------------
IMG_PATH = Path("scorebug_test.png")

UPSCALE = 2               # 1–3 usually enough
CLAHE = True              # local contrast helps the dark slab

# Edge/Dilate/Clean
SCHARR_SCALE = 1          # leave as 1
DILATE_W, DILATE_H = 3, 3 # make edges solid
CLOSE_W,  CLOSE_H  = 5, 3 # bridge gaps inside digits
OPEN_W,   OPEN_H   = 3, 3 # remove salt after closing (0=skip if you set any to 0)

# Left strip to ignore (flag)
EXCLUDE_LEFT_FRAC = 0.14  # 0.10–0.20

# Filtering (very forgiving; tighten later)
MIN_AREA = 24
MAX_AREA_FRAC = 0.35

MIN_ASPECT = 0.35         # w/h lower bound (allow narrow "1")
MAX_ASPECT = 18.0         # exclude ultra-flat chrome lines

# Row-height model (derived from data)
ROW_H_LOW_FRAC  = 0.55    # keep if h is within [low*row_h, high*row_h]
ROW_H_HIGH_FRAC = 1.45

# Fill ratio – how solid the blob must be
MIN_FILL = 0.12
MAX_FILL = 0.95

# Border-touching
DROP_BORDER_TOUCHING = False
# -------------------------------------------------------------

def imread_gray(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    if UPSCALE > 1:
        h, w = im.shape[:2]
        im = cv2.resize(im, (w*UPSCALE, h*UPSCALE), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
    return g

def scharr_edges(g):
    sx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Otsu on edges -> binary edge mask
    _, bw = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw, mag

def thicken_clean(bw):
    m = bw.copy()
    if DILATE_W and DILATE_H:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_W, DILATE_H))
        m = cv2.dilate(m, k, iterations=1)
    if CLOSE_W and CLOSE_H:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_W, CLOSE_H))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if OPEN_W and OPEN_H:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_W, OPEN_H))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m

def external_contours(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def contour_items(cnts, shape):
    H, W = shape
    items = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = int(cv2.contourArea(c))
        touch = (x <= 0 or y <= 0 or x+w >= W-1 or y+h >= H-1)
        items.append({"bbox":(x,y,w,h),"area":area,"touch":touch})
    return items

def estimate_row_height(items, W, H):
    # Use the median of the top third heights (robust to clutter)
    hs = sorted([h for (_,_,_,h) in (it["bbox"] for it in items)])
    if not hs: return max(8, H//6)  # fallback
    k = max(1, len(hs)//3)
    return int(np.median(hs[-k:]))

def filter_items(items, img_size):
    W, H = img_size
    left_cut = int(EXCLUDE_LEFT_FRAC * W)
    # estimate row height from candidates
    row_h = estimate_row_height(items, W, H)

    kept = []
    for it in items:
        x,y,w,h = it["bbox"]
        area = it["area"]
        if x + w <= left_cut: continue
        if area < MIN_AREA or area > MAX_AREA_FRAC*(W*H): continue
        if DROP_BORDER_TOUCHING and it["touch"]: continue

        if not (ROW_H_LOW_FRAC*row_h <= h <= ROW_H_HIGH_FRAC*row_h):
            continue

        aspect = w / float(max(1,h))
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT): continue

        fill = area / float(max(1,w*h))
        if not (MIN_FILL <= fill <= MAX_FILL): continue

        kept.append(it)
    return kept

def draw_boxes(base_gray, items, color, label, out_path):
    rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    for it in items:
        x,y,w,h = it["bbox"]
        cv2.rectangle(rgb, (x,y), (x+w, y+h), color, 2)
    cv2.putText(rgb, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, rgb)

def main():
    g = imread_gray(IMG_PATH)

    edge_bin, edge_mag = scharr_edges(g)
    cv2.imwrite("dbg_edges_mag.png", edge_mag)
    cv2.imwrite("dbg_edges_bin.png", edge_bin)

    thick = thicken_clean(edge_bin)
    cv2.imwrite("dbg_edges_solid.png", thick)

    cnts = external_contours(thick)
    items_all = contour_items(cnts, g.shape)
    draw_boxes(g, items_all, (0,160,0), "ALL (external contours)", "dbg_all_boxes.png")

    kept = filter_items(items_all, (g.shape[1], g.shape[0]))
    draw_boxes(g, kept, (0,255,255), "KEPT", "dbg_kept_boxes.png")

    print(f"contours: total={len(items_all)} kept={len(kept)}")

if __name__ == "__main__":
    main()
