# sweep_edges.py
import cv2, numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime

# ----------------- INPUT -----------------
IMG_PATH = Path("scorebug_test.png")
OUT_DIR  = Path("sweep_out")
UPSCALE  = 2
USE_CLAHE = True
# -----------------------------------------

def to_bgr(img):
    if img is None:
        raise ValueError("to_bgr got None")
    if len(img.shape) == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def draw_boxes(gray, boxes, color, label):
    rgb = to_bgr(gray.copy())
    for (x,y,w,h) in boxes:
        cv2.rectangle(rgb, (x,y), (x+w,y+h), color, 2)
    cv2.putText(rgb, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return rgb

def make_sheet(title, *imgs):
    # make all BGR, same height, padded, then hconcat
    imgs = [to_bgr(i) for i in imgs]
    H = max(i.shape[0] for i in imgs)
    imgs = [cv2.resize(i, (int(i.shape[1]*H/i.shape[0]), H), interpolation=cv2.INTER_AREA) for i in imgs]
    pad = lambda a: cv2.copyMakeBorder(a, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(30,30,30))
    row = cv2.hconcat([pad(i) for i in imgs])
    cv2.putText(row, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    return row

def load_gray():
    im = cv2.imread(str(IMG_PATH), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(IMG_PATH)
    if UPSCALE > 1:
        h,w = im.shape[:2]
        im = cv2.resize(im, (w*UPSCALE, h*UPSCALE), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    return g

def scharr_edges(g):
    sx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mag, bw

def thicken_clean(bw, dil_w, dil_h, close_w, close_h, open_w, open_h):
    m = bw.copy()
    if dil_w and dil_h:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)), iterations=1)
    if close_w and close_h:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)), iterations=1)
    if open_w and open_h:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, open_h)), iterations=1)
    return m

def find_external_boxes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts]

def filter_boxes(boxes, W, H, *,
                 excl_left_frac,
                 min_area,
                 max_area_frac,
                 min_aspect,
                 max_aspect,
                 row_h_low,
                 row_h_high,
                 min_fill,
                 max_fill,
                 drop_touch=False):
    left_cut = int(excl_left_frac * W)
    # estimate row height = median of top third heights
    hs = sorted([h for (_,_,_,h) in boxes])
    row_h = (np.median(hs[-max(1,len(hs)//3):]) if hs else H/6)

    kept = []
    area_img = W*H
    for (x,y,w,h) in boxes:
        if x+w <= left_cut: continue
        area = w*h
        if area < min_area or area > max_area_frac*area_img: continue

        aspect = w / max(1,h)
        if not (min_aspect <= aspect <= max_aspect): continue

        # crude fill from mask crop later; for now approximate by contour box area (ok for sweep)
        # fill gates are lenient already; skip if wildly outside
        fill_est = 0.5
        if not (min_fill <= fill_est <= max_fill): continue

        # height band vs row_h
        if not (row_h_low*row_h <= h <= row_h_high*row_h): continue

        # border touch test (optional)
        if drop_touch and (x<=0 or y<=0 or x+w>=W-1 or y+h>=H-1): continue

        kept.append((x,y,w,h))
    return kept

def run_variant(g, knobs):
    mag, edge_bin = scharr_edges(g)
    thick = thicken_clean(edge_bin, knobs["DIL_W"], knobs["DIL_H"], knobs["CLS_W"], knobs["CLS_H"], knobs["OPN_W"], knobs["OPN_H"])
    boxes_all = find_external_boxes(thick)

    H, W = g.shape[:2]
    boxes_kept = filter_boxes(
        boxes_all, W, H,
        excl_left_frac = knobs["EXCL_LEFT"],
        min_area       = knobs["MIN_AREA"],
        max_area_frac  = knobs["MAX_AREA_FRAC"],
        min_aspect     = knobs["MIN_ASPECT"],
        max_aspect     = knobs["MAX_ASPECT"],
        row_h_low      = knobs["ROW_LOW"],
        row_h_high     = knobs["ROW_HIGH"],
        min_fill       = knobs["MIN_FILL"],
        max_fill       = knobs["MAX_FILL"],
        drop_touch     = knobs["DROP_TOUCH"]
    )

    img_all  = draw_boxes(g, boxes_all,  (0,160,0), "ALL")
    img_kept = draw_boxes(g, boxes_kept, (0,255,255), "KEPT")

    sheet = make_sheet(
        f"{knobs}",
        to_bgr(g), to_bgr(mag), to_bgr(edge_bin), to_bgr(thick), img_all, img_kept
    )
    return sheet, len(boxes_all), len(boxes_kept)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_gray()

    # ------- parameter grid (kept small but useful) -------
    DILS   = [(2,3), (3,4), (4,5)]
    CLOSES = [(5,2), (7,3), (9,3)]
    OPENS  = [(0,0), (2,2)]
    EXCLS  = [0.10, 0.14, 0.18]
    ASP    = [(0.30, 18.0), (0.40, 16.0)]
    ROWB   = [(0.55, 1.60), (0.60, 1.50)]
    FILL   = [(0.08, 0.95), (0.12, 0.90)]
    AREA   = [24, 36]
    MAXAF  = [0.35]

    grid = list(product(DILS, CLOSES, OPENS, EXCLS, ASP, ROWB, FILL, AREA, MAXAF))
    print(f"variants to test: {len(grid)}")

    idx = 0
    for (dil, clos, opn, excl, asp, rowb, fillb, min_area, max_area_frac) in grid:
        knobs = dict(
            DIL_W=dil[0], DIL_H=dil[1],
            CLS_W=clos[0], CLS_H=clos[1],
            OPN_W=opn[0],  OPN_H=opn[1],
            EXCL_LEFT=excl,
            MIN_ASPECT=asp[0], MAX_ASPECT=asp[1],
            ROW_LOW=rowb[0], ROW_HIGH=rowb[1],
            MIN_FILL=fillb[0], MAX_FILL=fillb[1],
            MIN_AREA=min_area, MAX_AREA_FRAC=max_area_frac,
            DROP_TOUCH=False
        )
        try:
            sheet, n_all, n_kept = run_variant(g, knobs)
        except Exception as e:
            print(f"[{idx:03}] ERROR {e}  knobs={knobs}")
            idx += 1
            continue

        fn = OUT_DIR / f"sweep_{idx:03}_all{n_all}_kept{n_kept}.png"
        ok = cv2.imwrite(str(fn), sheet)
        if not ok:
            print(f"[{idx:03}] FAILED TO WRITE {fn}")
        else:
            print(f"[{idx:03}] wrote {fn}  (all={n_all}, kept={n_kept})")
        idx += 1

    print("Done.")

if __name__ == "__main__":
    main()
