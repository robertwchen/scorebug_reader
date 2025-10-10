#!/usr/bin/env python3
# tennis_scorebug_pipeline.py
# End-to-end: detect names, digits, serving dot; OCR; infer tennis state (incl. tiebreak vs game points)
# Usage: python tennis_scorebug_pipeline.py [image_path]

import cv2, numpy as np, string, json, sys
from pathlib import Path

# ---------------- Config ----------------
UPSCALE = 2
LEFT_CUT_FRAC = 0.12
ROW_H_ALPHA   = 2.2

# Digit MSER filters
ASPECT_MIN_NUM, ASPECT_MAX_NUM = 0.28, 1.25
HEIGHT_MIN_FR_NUM, HEIGHT_MAX_FR_NUM = 0.35, 1.35
AREA_MIN_NUM = 35
AREA_MAX_FR_NUM = 0.25

# Name band filters
ASPECT_MIN_NAME = 1.6
ASPECT_MAX_NAME = 20.0
HEIGHT_MIN_FR_NAME, HEIGHT_MAX_FR_NAME = 0.40, 1.45
AREA_MIN_NAME = 120
AREA_MAX_FR_NAME = 0.35
NAME_BAND_FRAC = (0.12, 0.65)

# Serving-dot HSV and geometry
GREEN_LOWER = (45, 150, 120)
GREEN_UPPER = (85, 255, 255)
DOT_X_BAND  = (0.55, 0.72)  # relative x range where dot lives (between name and first set digits)
KILL_RIGHT_X = 0.80         # ignore far-right green tiles

# ---------------- IO helpers ----------------
def imread_gray(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    if UPSCALE>1:
        h,w = im.shape[:2]
        im = cv2.resize(im, (w*UPSCALE, h*UPSCALE), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    return im, g

# ---------------- Detection ----------------
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
                    out[i] = (nx1,ny1,nx2-nx1,ny2-ny1); keep=False; break
        if keep: out.append((x,y,w,h))
    return out

def filter_digitish(rects, W, H):
    left_cut = int(LEFT_CUT_FRAC * W)
    approx_row_h = H / ROW_H_ALPHA
    mid_x1, mid_x2 = int(0.52*W), int(0.89*W)
    right_x1 = int(0.89*W)

    kept = []
    for (x,y,w,h) in rects:
        if x+w <= left_cut: 
            continue
        if not ( (mid_x1 <= x <= mid_x2) or (x >= right_x1) ):
            continue
        area = w*h
        if area < AREA_MIN_NUM or area > AREA_MAX_FR_NUM*W*H:
            continue
        ar = w/float(h)
        if ar < ASPECT_MIN_NUM or ar > ASPECT_MAX_NUM:
            continue
        if not (HEIGHT_MIN_FR_NUM*approx_row_h <= h <= HEIGHT_MAX_FR_NUM*approx_row_h):
            continue
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

def detect_serving_dots(im, name_boxes):
    H,W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower = np.array(GREEN_LOWER, dtype=np.uint8)
    upper = np.array(GREEN_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask[:, int(KILL_RIGHT_X*W):] = 0  # kill far-right green tiles
    mask = cv2.medianBlur(mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    approx_row_h = H / ROW_H_ALPHA
    min_area, max_area = 0.02*(approx_row_h**2), 0.55*(approx_row_h**2)
    band_x1, band_x2 = int(DOT_X_BAND[0]*W), int(DOT_X_BAND[1]*W)

    if name_boxes:
        centers = sorted([y+h/2 for (_,y,_,h) in name_boxes])[:2]
        if len(centers) == 1:
            centers = [centers[0]-approx_row_h/1.3, centers[0]+approx_row_h/1.3]
    else:
        centers = [H/3, 2*H/3]

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

# ---------------- OCR ----------------
def try_import_tesseract():
    import importlib.util
    spec = importlib.util.find_spec("pytesseract")
    if spec is None:
        return None
    import pytesseract
    return pytesseract

def render_char(ch, size=32, thickness=2):
    canvas = np.zeros((size, size), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts = 0.9 if ch.isdigit() else 0.8
    (tw, th), _ = cv2.getTextSize(ch, font, ts, thickness)
    x = (size - tw)//2; y = (size + th)//2
    cv2.putText(canvas, ch, (x,y), font, ts, 255, thickness, cv2.LINE_AA)
    return canvas

DIGITS_TMPL = {str(d): render_char(str(d)) for d in range(10)}
LETTERS_TMPL = {c: render_char(c) for c in string.ascii_uppercase}

def binarize_tile(tile):
    g = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim==3 else tile.copy()
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(bw) < 127: bw = 255 - bw
    return bw

def match_char(bw, templates):
    x,y,w,h = cv2.boundingRect((bw>0).astype(np.uint8))
    crop = bw[y:y+h, x:x+w]
    if min(crop.shape[:2]) <= 0:
        return None, -1.0
    sq = np.zeros((32,32), np.uint8)
    scale = 28.0 / max(crop.shape)
    rs = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    padx = (32 - rs.shape[1])//2; pady = (32 - rs.shape[0])//2
    sq[pady:pady+rs.shape[0], padx:padx+rs.shape[1]] = rs

    best, best_score = None, -1.0
    for ch, tmpl in templates.items():
        res = cv2.matchTemplate(sq, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(res[0][0])
        if score > best_score:
            best_score, best = score, ch
    return best, best_score

def ocr_digit_crop(img, pyt):
    if pyt is not None:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3,3), 0)
        _, bw = cv2.threshold(g, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        outs = []
        for pol in [bw, 255-bw]:
            cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
            s = pyt.image_to_string(pol, config=cfg).strip()
            outs.append(s)
        s = max(outs, key=lambda x: sum(ch.isdigit() for ch in x))
        s = "".join([ch for ch in s if ch.isdigit()])
        if s.isdigit():
            return int(s)
    # fallback template
    bw = binarize_tile(img)
    ch, sc = match_char(bw, DIGITS_TMPL)
    try: return int(ch) if ch is not None else None
    except: return None

def ocr_name_crop(img, pyt):
    if pyt is not None:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3,3), 0)
        bw = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
        texts = []
        for pol in [bw, 255-bw]:
            cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            s = pyt.image_to_string(pol, config=cfg).strip().upper()
            texts.append(s)
        s = max(texts, key=lambda x: sum(ch in string.ascii_uppercase for ch in x))
        s = "".join([ch for ch in s if ch in string.ascii_uppercase])
        if len(s) >= 3:
            return s
    # fallback template segmentation
    h, w = img.shape[:2]
    bw = binarize_tile(img)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letters = []
    for c in cnts:
        X,Y,W,H = cv2.boundingRect(c)
        if H < h*0.35 or W < 3: 
            continue
        if W/H > 1.5 and W > 0.25*w:
            continue
        letters.append((X,Y,W,H))
    letters.sort(key=lambda b: b[0])
    text = ""
    for (X,Y,W,H) in letters:
        tile = bw[Y:Y+H, X:X+W]
        ch, sc = match_char(tile, LETTERS_TMPL)
        text += ch if ch else ""
    text = "".join([c for c in text if c in string.ascii_uppercase])
    return text

# ---------------- State logic ----------------
def infer_tennis_state(set_cols, right_col, server_row):
    def is_completed_set(a,b):
        hi, lo = max(a,b), min(a,b)
        if hi < 6: return False
        if hi >= 6 and hi - lo >= 2: return True
        if hi == 7 and lo == 6: return True
        return False

    sets_won_top = sets_won_bot = 0
    last_completed_idx = -1
    for i,(t,b) in enumerate(set_cols):
        if is_completed_set(t,b):
            last_completed_idx = i
            if t>b: sets_won_top += 1
            else:   sets_won_bot += 1
        else:
            break

    current_set_idx = last_completed_idx if last_completed_idx == len(set_cols)-1 else last_completed_idx + 1

    is_tb = False
    game_score = {'type':'none','score':None}
    ok = True; reason = ''

    def map_int_points_to_tennis(p):
        if p in (0, 15, 30, 40): return str(p)
        if p in ('Ad','A'): return 'Ad'
        if isinstance(p,int):
            return ['0','15','30','40','Ad'][p] if 0 <= p <= 4 else str(p)
        return str(p)

    live_set = set_cols[current_set_idx] if 0 <= current_set_idx < len(set_cols) else None
    if live_set and live_set[0] == 6 and live_set[1] == 6 and isinstance(right_col, tuple):
        a,b = right_col
        if all(isinstance(x,int) and x >= 0 for x in (a,b)):
            is_tb = True
            game_score = {'type':'tiebreak','score':(a,b)}
        else:
            ok, reason = False, 'Expected integer tiebreak points at 6–6.'
    else:
        if right_col is None:
            game_score = {'type':'points','score':None}
        elif isinstance(right_col, tuple):
            p_top = map_int_points_to_tennis(right_col[0])
            p_bot = map_int_points_to_tennis(right_col[1])
            game_score = {'type':'points','score':(p_top,p_bot)}
        else:
            game_score = {'type':'points','score':right_col}

    return {
        'currentSetIndex': current_set_idx,
        'setsWonTop'     : sets_won_top,
        'setsWonBottom'  : sets_won_bot,
        'perSetScores'   : list(set_cols),
        'isTiebreak'     : is_tb,
        'gameScore'      : game_score,
        'server'         : 'top' if server_row == 0 else 'bottom',
        'ok'             : ok,
        'reason'         : reason
    }

# ---------------- Pipeline ----------------
def run_pipeline(image_path: Path, result_json: Path=None, overlay_path: Path=None):
    im, g = imread_gray(image_path)
    H,W = g.shape[:2]

    # detect
    mser_rects = nms_union(mser_boxes(g), 0.3)
    digit_boxes = nms_union(filter_digitish(mser_rects, W, H), 0.2)
    name_boxes, _ = find_name_boxes(g)
    serve_dots, _ = detect_serving_dots(im, name_boxes)

    # names
    pyt = try_import_tesseract()
    name_boxes_sorted = sorted(name_boxes, key=lambda b: b[1])
    names = []
    for b in name_boxes_sorted:
        x,y,w,h = b
        crop = im[y:y+h, x:x+w]
        names.append(ocr_name_crop(crop, pyt))

    # digits -> columns
    if digit_boxes:
        med_y = np.median([y+h/2 for (_,y,_,h) in digit_boxes])
    else:
        med_y = H/2
    items = []
    for (x,y,w,h) in digit_boxes:
        crop = im[y:y+h, x:x+w]
        val  = ocr_digit_crop(crop, pyt)
        row  = 0 if (y+h/2) <= med_y else 1
        items.append(((x,y,w,h), val, row))

    cols = {}
    for (x,y,w,h), val, row in items:
        if val is None: continue
        xc = x + w/2
        found = None
        for k in list(cols.keys()):
            if abs(k - xc) < W*0.02:
                found = k; break
        if found is None:
            cols[xc] = [None, None]; found = xc
        cols[found][row] = val

    sorted_cols = [tuple(cols[k]) for k in sorted(cols.keys())]
    set_cols  = sorted_cols[:-1] if len(sorted_cols)>=3 else sorted_cols
    right_col = sorted_cols[-1]  if len(sorted_cols)>=3 else None

    server_row = serve_dots[0]['row'] if serve_dots else 0

    state = infer_tennis_state(set_cols, right_col, server_row)

    result = {
        "players": {
            "top": names[0] if len(names)>0 else "",
            "bottom": names[1] if len(names)>1 else ""
        },
        "setScores": set_cols,
        "rightColumn": right_col,
        "serverRow": server_row,
        "state": state
    }

    # Overlay
    if overlay_path:
        vis = im.copy()
        for (x,y,w,h) in digit_boxes:
            cv2.rectangle(vis, (x,y),(x+w,y+h), (0,255,255), 2)
        for (x,y,w,h) in name_boxes_sorted:
            cv2.rectangle(vis, (x,y),(x+w,y+h), (255,0,255), 2)
        for d in serve_dots:
            x,y,w,h = d['bbox']
            cv2.rectangle(vis, (x,y),(x+w,y+h), (0,255,0), 2)
            cv2.putText(vis, f"S{d['row']}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imwrite(str(overlay_path), vis)

    if result_json:
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2)

    return result

def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scorebug_test.png")
    result = run_pipeline(img_path, Path("tennis_scorebug_result.json"), Path("tennis_scorebug_overlay.png"))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
