# combine_names_numbers.py
# MSER for digits; morphology/MSER for player name blocks; HSV blob for serving dot.
import cv2, numpy as np
from pathlib import Path

IMG_PATH = Path("scorebug_test.png")   # set to your path
UPSCALE = 2

# Shared knobs
LEFT_CUT_FRAC = 0.12
ROW_H_ALPHA   = 2.2

# Number (digit tile) filters (from MSER approach)
ASPECT_MIN_NUM, ASPECT_MAX_NUM = 0.28, 1.25
HEIGHT_MIN_FR_NUM, HEIGHT_MAX_FR_NUM = 0.35, 1.35
AREA_MIN_NUM = 35
AREA_MAX_FR_NUM = 0.25

# Name block filters (morphology/MSER approach)
ASPECT_MIN_NAME = 1.6
ASPECT_MAX_NAME = 20.0
HEIGHT_MIN_FR_NAME, HEIGHT_MAX_FR_NAME = 0.40, 1.45
AREA_MIN_NAME = 120
AREA_MAX_FR_NAME = 0.35
NAME_BAND_FRAC = (0.12, 0.65)   # horizontal range where names live

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
    mser = cv2.MSER_create()
    mser.setDelta(3); mser.setMinArea(25); mser.setMaxArea(8000)
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
        a = w*h; keep = True
        for i,(X,Y,W,H) in enumerate(out):
            A = W*H
            ix1,iy1 = max(x,X), max(y,Y)
            ix2,iy2 = min(x+w, X+W), min(y+h, Y+H)
            iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
            inter = iw*ih
            if inter>0:
                iou = inter/float(a+A-inter)
                if iou >= iou_thr:
                    nx1,ny1 = min(x,X), min(y,Y)
                    nx2,ny2 = max(x+w, X+W), max(y+h, Y+H)
                    out[i] = (nx1,ny1,nx2-nx1,ny2-ny1); keep = False; break
        if keep: out.append((x,y,w,h))
    return out

def filter_digitish(rects, W, H):
    left_cut = int(LEFT_CUT_FRAC * W)
    approx_row_h = H / ROW_H_ALPHA
    mid_x1, mid_x2 = int(0.52*W), int(0.89*W)
    right_x1 = int(0.89*W)

    kept = []
    for (x,y,w,h) in rects:
        if x+w <= left_cut: continue
        if not ( (mid_x1 <= x <= mid_x2) or (x >= right_x1) ): continue
        area = w*h
        if area < AREA_MIN_NUM or area > AREA_MAX_FR_NUM*W*H: continue
        ar = w/float(h)
        if ar < ASPECT_MIN_NUM or ar > ASPECT_MAX_NUM: continue
        if not (HEIGHT_MIN_FR_NUM*approx_row_h <= h <= HEIGHT_MAX_FR_NUM*approx_row_h): continue
        kept.append((x,y,w,h))
    return kept

def find_name_boxes(g):
    H,W = g.shape[:2]
    approx_row_h = H / ROW_H_ALPHA
    left_cut = int(LEFT_CUT_FRAC * W)
    name_x1 = int(NAME_BAND_FRAC[0]*W)
    name_x2 = int(NAME_BAND_FRAC[1]*W)

    mser = cv2.MSER_create()
    mser.setDelta(3); mser.setMinArea(20); mser.setMaxArea(6000)

    rects = []
    for img in (g, 255-g):
        regs, _ = mser.detectRegions(img)
        for r in regs:
            x,y,w,h = cv2.boundingRect(r.reshape(-1,1,2))
            if x+w <= left_cut: continue
            if x < name_x1 or x > name_x2: continue
            area = w*h
            if area < 20: continue
            ar = w/float(h)
            if ar < 0.15 or ar > 5.0: continue
            if not (0.35*approx_row_h <= h <= 1.40*approx_row_h): continue
            rects.append((x,y,w,h))

    if not rects: return [], np.zeros_like(g)

    ycs = np.array([y+h/2 for (_,y,_,h) in rects], dtype=np.float32)
    med = np.median(ycs)
    top = [b for b in rects if (b[1]+b[3]/2.0) <= med]
    bot = [b for b in rects if (b[1]+b[3]/2.0) >  med]

    boxes = []
    for group in (top, bot):
        if not group: continue
        xs1 = [x for (x,_,_,_) in group]
        xs2 = [x+w for (x,_,w,_) in group]
        ys1 = [y for (_,y,_,_) in group]
        ys2 = [y+h for (_,y,_,h) in group]
        x1, x2 = max(name_x1, min(xs1)-6), min(name_x2, max(xs2)+6)
        y1, y2 = max(0, min(ys1)-4), min(H-1, max(ys2)+4)
        ww, hh = x2-x1, y2-y1
        if ww*hh < AREA_MIN_NAME: continue
        ar = ww/float(hh)
        if ar < ASPECT_MIN_NAME or ar > ASPECT_MAX_NAME: continue
        boxes.append((int(x1), int(y1), int(ww), int(hh)))

    mask = np.zeros_like(g)
    for (x,y,w,h) in rects:
        mask[y:y+h, x:x+w] = 255
    return boxes, mask

# ---------- Serving dot ----------
def detect_serving_dots(im, name_boxes):
    H,W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # bright tennis-green
    lower = np.array([45, 150, 120], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # kill far-right green tiles
    mask[:, int(0.80*W):] = 0

    mask = cv2.medianBlur(mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approx_row_h = H / ROW_H_ALPHA
    min_area, max_area = 0.02*(approx_row_h**2), 0.55*(approx_row_h**2)
    band_x1, band_x2 = int(0.55*W), int(0.72*W)

    if name_boxes:
        centers = sorted([y+h/2 for (_,y,_,h) in name_boxes])[:2]
        if len(centers) == 1:
            centers = [centers[0]-approx_row_h/1.3, centers[0]+approx_row_h/1.3]
    else:
        centers = [H/3, 2*H/3]

    results = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if x < band_x1 or (x+w) > band_x2: continue
        if area < min_area or area > max_area: continue
        a = cv2.contourArea(c); p = cv2.arcLength(c, True)
        if p == 0: continue
        circ = 4*np.pi*a/(p*p)
        if circ < 0.6: continue
        yc = y + h/2
        row = 0 if abs(yc - centers[0]) <= abs(yc - centers[-1]) else 1
        results.append({'bbox': (int(x),int(y),int(w),int(h)), 'row': row})

    results.sort(key=lambda d: d['bbox'][0])
    return results, mask

def draw(im, rects, color, label, fn):
    vis = im.copy()
    for (x,y,w,h) in rects:
        cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
    cv2.putText(vis, label, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(fn, vis)

def main():
    im, g = imread_gray(IMG_PATH)
    H,W = g.shape[:2]

    # numbers
    mser_rects = mser_boxes(g)
    mser_rects = nms_union(mser_rects, 0.3)
    digit_boxes = filter_digitish(mser_rects, W, H)
    digit_boxes = nms_union(digit_boxes, 0.2)

    # names
    name_boxes, name_band = find_name_boxes(g)

    # serving dot
    serve_dots, green_mask = detect_serving_dots(im, name_boxes)

    # save debugs
    cv2.imwrite("names_band_mask.png", name_band)
    cv2.imwrite("serve_green_mask.png", green_mask)

    # union visualization
    vis = im.copy()
    for (x,y,w,h) in digit_boxes:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 2)
    for (x,y,w,h) in name_boxes:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,255), 2)
    for d in serve_dots:
        x,y,w,h = d['bbox']
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(vis, f"S{d['row']}", (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"TOTAL BOXES: {len(digit_boxes)+len(name_boxes)}",
                (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.imwrite("combined_all.png", vis)

    print("digits:", len(digit_boxes), digit_boxes)
    print("names :", len(name_boxes),  name_boxes)
    print("serve_dots:", [{'bbox': d['bbox'], 'row': d['row']} for d in serve_dots])

if __name__ == "__main__":
    main()
