
import cv2, numpy as np
from pathlib import Path

IMG_PATH = Path("scorebug_test.png")
UPSCALE = 2

LEFT_CUT_FRAC = 0.12
ROW_H_ALPHA   = 2.2
ASPECT_MIN, ASPECT_MAX = 0.28, 1.25
HEIGHT_MIN_FR, HEIGHT_MAX_FR = 0.35, 1.35
AREA_MIN = 35
AREA_MAX_FR = 0.25

def imread_gray(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    if UPSCALE>1:
        h,w = im.shape[:2]
        im = cv2.resize(im, (w*UPSCALE, h*UPSCALE), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    return im, g

def mser_boxes(gray):
    mser = cv2.MSER_create()   # defaults
    # Tune a bit:
    mser.setDelta(3)
    mser.setMinArea(25)
    mser.setMaxArea(8000)
    regions, _ = mser.detectRegions(gray)
    regions_inv, _ = mser.detectRegions(255-gray)
    rects = []
    for regs in (regions, regions_inv):
        for r in regs:
            x,y,w,h = cv2.boundingRect(r.reshape(-1,1,2))
            rects.append((x,y,w,h))
    return rects

def nms_union(rects, iou_thr=0.25):
    out = []
    for x,y,w,h in sorted(rects, key=lambda b:(b[0],b[1])):
        a = w*h
        keep = True
        for i,(X,Y,W,H) in enumerate(out):
            A = W*H
            ix1,iy1 = max(x,X), max(y,Y)
            ix2,iy2 = min(x+w, X+W), min(y+h, Y+H)
            iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
            inter = iw*ih
            if inter>0:
                iou = inter/float(a+A-inter)
                if iou >= iou_thr:
                    nx1, ny1 = min(x,X), min(y,Y)
                    nx2, ny2 = max(x+w, X+W), max(y+h, Y+H)
                    out[i] = (nx1, ny1, nx2-nx1, ny2-ny1)
                    keep = False
                    break
        if keep:
            out.append((x,y,w,h))
    return out


def filter_digitish(rects, W, H):
    left_cut = int(LEFT_CUT_FRAC * W)
    approx_row_h = H / ROW_H_ALPHA

    # define horizontal bands: mid sets and far-right column
    mid_x1, mid_x2 = int(0.52*W), int(0.89*W)
    right_x1 = int(0.89*W)

    kept = []
    for (x,y,w,h) in rects:
        if x+w <= left_cut:
            continue
        # only keep if in mid band or rightmost column
        if not ( (mid_x1 <= x <= mid_x2) or (x >= right_x1) ):
            continue

        area = w*h
        if area < AREA_MIN or area > AREA_MAX_FR*W*H:
            continue
        ar = w/float(h)
        if ar < ASPECT_MIN or ar > ASPECT_MAX:
            continue
        if not (HEIGHT_MIN_FR*approx_row_h <= h <= HEIGHT_MAX_FR*approx_row_h):
            continue
        kept.append((x,y,w,h))
    return kept

def draw(im, rects, color, label, fn):
    vis = im.copy()
    for (x,y,w,h) in rects:
        cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
    cv2.putText(vis, label, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(fn, vis)

def main():
    im, g = imread_gray(IMG_PATH)
    H,W = g.shape[:2]

    raw = mser_boxes(g)
    raw = nms_union(raw, 0.3)
    kept = filter_digitish(raw, W, H)
    kept = nms_union(kept, 0.2)

    draw(im, raw,  (0,160,0), "ALL_MSER", "mser_all.png")
    draw(im, kept, (0,255,255), f"KEPT ({len(kept)})", "mser_kept.png")

    print("kept:", len(kept))
    print("boxes:", kept)

if __name__ == "__main__":
    main()
