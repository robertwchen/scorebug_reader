#!/usr/bin/env python3
# MSER numbers + names (your logic), robust server-marker detection, batch mode,
# and per-image segmentation sheets.

import cv2, numpy as np, string, json, sys, glob
from pathlib import Path

# ---------------- Config (same defaults) ----------------
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

# -------- NEW: generalized server-marker detection --------
# Color families broadcasters use for bullets/arrows:
HSV_RANGES = [
    ((20,130,130),(42,255,255)),   # yellow
    ((42,120,120),(90,255,255)),   # lime/green
    ((10,130,130),(20,255,255)),   # orange-ish
]
# Typical horizontal zone between name and first set column:
SERVER_GAP_BAND = (0.48, 0.82)     # a little wider than before
# Also handle markers that appear JUST LEFT of the name (before-name arrow/dot):
LEFT_OF_NAME_PAD = 0.06            # +/- width fraction around name_x1 (per row)
# Size prior relative to row-height^2:
SERVER_SIZE_FR  = (0.02, 0.80)     # min/max
MIN_SCORE_TO_KEEP = 0.34           # keep slightly looser

SAVE_DEBUG_SHEETS = True

# ---------------- Small viz helpers ----------------
def to_bgr(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim==2 else img
def put_label(img, text, color=(0,255,0)):
    im = to_bgr(img.copy()); cv2.putText(im, text, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA); return im
def rect(im, b, color, t=2): x,y,w,h=b; cv2.rectangle(im,(x,y),(x+w,y+h),color,t); return im
def draw_boxes(img, boxes, color, thick=2):
    im=img.copy()
    for b in boxes: rect(im,b,color,thick)
    return im
def make_sheet(*rows, pad=6, bg=(24,24,24)):
    prepared=[]
    for row in rows:
        row=[to_bgr(r) for r in row]
        H=max(r.shape[0] for r in row)
        row=[cv2.resize(r,(int(r.shape[1]*H/r.shape[0]),H), interpolation=cv2.INTER_AREA) for r in row]
        row=[cv2.copyMakeBorder(r,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=bg) for r in row]
        prepared.append(cv2.hconcat(row))
    return cv2.vconcat(prepared)

# ---------------- IO ----------------
def imread_gray(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    if UPSCALE>1:
        h,w = im.shape[:2]
        im = cv2.resize(im, (w*UPSCALE, h*UPSCALE), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    return im, g

# ---------------- MSER digits & names (your approach) ----------------
def mser_boxes(gray):
    mser = cv2.MSER_create()
    mser.setDelta(3); mser.setMinArea(25); mser.setMaxArea(8000)
    a,_ = mser.detectRegions(gray)
    b,_ = mser.detectRegions(255-gray)
    rects=[]
    for regs in (a,b):
        for r in regs:
            x,y,w,h = cv2.boundingRect(r.reshape(-1,1,2))
            rects.append((x,y,w,h))
    return rects

def nms_union(rects, iou_thr=0.25):
    out=[]
    for x,y,w,h in sorted(rects, key=lambda b:(b[0],b[1])):
        a=w*h; keep=True
        for i,(X,Y,W,H) in enumerate(out):
            A=W*H
            ix1,iy1=max(x,X),max(y,Y); ix2,iy2=min(x+w,X+W),min(y+h,Y+H)
            iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
            if inter>0:
                iou=inter/float(a+A-inter)
                if iou>=iou_thr:
                    out[i]=(min(x,X),min(y,Y), max(x+w,X+W)-min(x,X), max(y+h,Y+H)-min(y,Y))
                    keep=False; break
        if keep: out.append((x,y,w,h))
    return out

def filter_digitish(rects, W, H):
    left_cut=int(LEFT_CUT_FRAC*W); approx_row_h=H/ROW_H_ALPHA
    mid_x1, mid_x2 = int(0.52*W), int(0.89*W); right_x1=int(0.89*W)
    kept=[]
    for (x,y,w,h) in rects:
        if x+w<=left_cut: continue
        if not ((mid_x1<=x<=mid_x2) or (x>=right_x1)): continue
        area=w*h
        if area<AREA_MIN_NUM or area>AREA_MAX_FR_NUM*W*H: continue
        ar=w/float(h)
        if ar<ASPECT_MIN_NUM or ar>ASPECT_MAX_NUM: continue
        if not (HEIGHT_MIN_FR_NUM*approx_row_h <= h <= HEIGHT_MAX_FR_NUM*approx_row_h): continue
        kept.append((x,y,w,h))
    return kept

def find_name_boxes(g):
    H,W=g.shape[:2]; approx_row_h=H/ROW_H_ALPHA
    left_cut=int(LEFT_CUT_FRAC*W); name_x1=int(NAME_BAND_FRAC[0]*W); name_x2=int(NAME_BAND_FRAC[1]*W)
    mser=cv2.MSER_create(); mser.setDelta(3); mser.setMinArea(20); mser.setMaxArea(6000)
    rects=[]
    for img in (g,255-g):
        regs,_=mser.detectRegions(img)
        for r in regs:
            x,y,w,h=cv2.boundingRect(r.reshape(-1,1,2))
            if x+w<=left_cut or x<name_x1 or x>name_x2: continue
            area=w*h
            if area<20: continue
            ar=w/float(h)
            if ar<0.15 or ar>5.0: continue
            if not (0.35*approx_row_h <= h <= 1.40*approx_row_h): continue
            rects.append((x,y,w,h))
    if not rects: return [], np.zeros_like(g)
    ycs=np.array([y+h/2 for (_,y,_,h) in rects], np.float32)
    med=float(np.median(ycs))
    top=[b for b in rects if (b[1]+b[3]/2.0)<=med]; bot=[b for b in rects if (b[1]+b[3]/2.0)>med]
    boxes=[]
    for group in (top,bot):
        if not group: continue
        xs1=[x for (x,_,_,_) in group]; xs2=[x+w for (x,_,w,_) in group]
        ys1=[y for (_,y,_,_) in group]; ys2=[y+h for (_,y,_,h) in group]
        x1,x2=max(name_x1, min(xs1)-6), min(name_x2, max(xs2)+6)
        y1,y2=max(0, min(ys1)-4), min(H-1, max(ys2)+4)
        ww,hh=x2-x1,y2-y1
        if ww*hh<AREA_MIN_NAME: continue
        ar=ww/float(hh)
        if ar<ASPECT_MIN_NAME or ar>ASPECT_MAX_NAME: continue
        boxes.append((int(x1),int(y1),int(ww),int(hh)))
    mask=np.zeros_like(g)
    return boxes, mask

# ---------------- Server-marker detector ----------------
def detect_server_markers(im, name_boxes):
    """Return [{'bbox':(x,y,w,h), 'row':0/1, 'score':float}, ...], plus debug masks."""
    H,W = im.shape[:2]
    approx_row_h = H / ROW_H_ALPHA

    # Preferred row centers (from name boxes if present)
    if name_boxes:
        centers = sorted([y+h/2 for (_,y,_,h) in name_boxes])[:2]
        if len(centers)==1:
            centers=[centers[0]-approx_row_h/1.3, centers[0]+approx_row_h/1.3]
    else:
        centers=[H/3, 2*H/3]

    # Two horizontal bands to search:
    #  (1) Between name and set columns
    band1 = (int(SERVER_GAP_BAND[0]*W), int(SERVER_GAP_BAND[1]*W))
    #  (2) Just left of each name (handles left-placed arrow/dot)
    left_masks = []
    for (x,y,w,h) in name_boxes:
        x1 = max(0, int(x - LEFT_OF_NAME_PAD*W))
        x2 = min(W, int(x + 0.02*W))
        left_masks.append((x1, x2, y, y+h))

    # Build color mask for both zones
    hsv=cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    color_mask=np.zeros((H,W), np.uint8)
    for lo,hi in HSV_RANGES:
        color_mask |= cv2.inRange(hsv, np.array(lo,np.uint8), np.array(hi,np.uint8))
    # keep only band1 and left-of-name rectangles
    keep = np.zeros_like(color_mask)
    keep[:, band1[0]:band1[1]] = 255
    for (x1,x2,y1,y2) in left_masks:
        keep[y1:y2, x1:x2] = 255
    color_mask = cv2.bitwise_and(color_mask, keep)
    color_mask = cv2.medianBlur(color_mask,3)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)

    # Also allow white/gray bullets (luminance)
    g=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g=cv2.GaussianBlur(g,(3,3),0)
    _, bright=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bright = cv2.bitwise_and(bright, keep)  # restrict to same zones
    bright=cv2.medianBlur(bright,3)

    cand=cv2.bitwise_or(color_mask, bright)
    contours,_=cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minA=SERVER_SIZE_FR[0]*(approx_row_h**2)
    maxA=SERVER_SIZE_FR[1]*(approx_row_h**2)

    scored=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c); a=w*h
        if a<minA or a>maxA: continue
        xc,yc=x+w/2.0, y+h/2.0

        area=cv2.contourArea(c); per=max(1.0, cv2.arcLength(c,True))
        circ=4*np.pi*area/(per*per)                # circle ~1.0
        ecc=max(w,h)/max(1.0,min(w,h))            # circle ~1.0
        ecc_score=max(0.0, 1.0-(ecc-1.0))
        approx=cv2.approxPolyDP(c, 0.12*per, True)
        poly_bonus=0.15 if (3<=len(approx)<=6) else 0.0

        # Which row is it closer to?
        dists=[abs(yc-cy) for cy in centers]; row=int(np.argmin(dists))
        row_score=max(0.0, 1.0-(dists[row]/(0.55*approx_row_h)))

        score=0.45*circ + 0.25*ecc_score + 0.20*row_score + 0.10*poly_bonus
        if score>=MIN_SCORE_TO_KEEP:
            scored.append({'bbox':(int(x),int(y),int(w),int(h)), 'row':row, 'score':float(score)})

    # keep best per row
    best={0:None,1:None}
    for s in sorted(scored, key=lambda d:d['score'], reverse=True):
        r=s['row']
        if best[r] is None:
            best[r]=s

    out=[best[0]] if best[0] else []
    if best[1]: out.append(best[1])

    return out, {'color_mask':color_mask, 'bright_mask':bright, 'cand':cand}

# ---------------- OCR helpers (same idea) ----------------
def try_import_tesseract():
    import importlib.util
    spec = importlib.util.find_spec("pytesseract")
    if spec is None: return None
    import pytesseract; return pytesseract

def render_char(ch, size=32, thickness=2):
    canvas=np.zeros((size,size), np.uint8)
    font=cv2.FONT_HERSHEY_SIMPLEX
    ts=0.9 if ch.isdigit() else 0.8
    (tw,th),_=cv2.getTextSize(ch,font,ts,thickness)
    x=(size-tw)//2; y=(size+th)//2
    cv2.putText(canvas,ch,(x,y),font,ts,255,thickness,cv2.LINE_AA)
    return canvas

DIGITS_TMPL={str(d):render_char(str(d)) for d in range(10)}
LETTERS_TMPL={c:render_char(c) for c in string.ascii_uppercase}

def binarize_tile(tile):
    g=cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim==3 else tile.copy()
    g=cv2.GaussianBlur(g,(3,3),0)
    _,bw=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(bw)<127: bw=255-bw
    return bw

def match_char(bw, templates):
    x,y,w,h=cv2.boundingRect((bw>0).astype(np.uint8))
    crop=bw[y:y+h, x:x+w]
    if min(crop.shape[:2])<=0: return None,-1.0
    sq=np.zeros((32,32), np.uint8)
    scale=28.0/max(crop.shape)
    rs=cv2.resize(crop,(int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    padx=(32-rs.shape[1])//2; pady=(32-rs.shape[0])//2
    sq[pady:pady+rs.shape[0], padx:padx+rs.shape[1]]=rs
    best,score=None,-1.0
    for ch,tmpl in templates.items():
        s=float(cv2.matchTemplate(sq, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
        if s>score: score, best=s, ch
    return best,score

def ocr_digit_crop(img, pyt):
    if pyt is not None:
        g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g=cv2.resize(g,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        g=cv2.GaussianBlur(g,(3,3),0)
        _,bw=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        outs=[]
        for pol in [bw,255-bw]:
            cfg="--psm 7 -c tessedit_char_whitelist=0123456789"
            outs.append(pyt.image_to_string(pol, config=cfg).strip())
        s=max(outs, key=lambda x: sum(ch.isdigit() for ch in x))
        s="".join([ch for ch in s if ch.isdigit()])
        if s.isdigit(): return int(s)
    bw=binarize_tile(img); ch,_=match_char(bw, DIGITS_TMPL)
    try: return int(ch) if ch is not None else None
    except: return None

def ocr_name_crop(img, pyt):
    if pyt is not None:
        g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g=cv2.resize(g,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        g=cv2.GaussianBlur(g,(3,3),0)
        bw=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
        texts=[]
        for pol in [bw,255-bw]:
            cfg="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            texts.append(pyt.image_to_string(pol, config=cfg).strip().upper())
        s=max(texts, key=lambda x: sum(ch in string.ascii_uppercase for ch in x))
        s="".join([ch for ch in s if ch in string.ascii_uppercase])
        if len(s)>=3: return s
    h,w=img.shape[:2]; bw=binarize_tile(img)
    cnts,_=cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letters=[]
    for c in cnts:
        X,Y,W,H=cv2.boundingRect(c)
        if H<h*0.35 or W<3: continue
        if W/H>1.5 and W>0.25*w: continue
        letters.append((X,Y,W,H))
    letters.sort(key=lambda b:b[0])
    text=""
    for (X,Y,W,H) in letters:
        ch,_=match_char(bw[Y:Y+H, X:X+W], LETTERS_TMPL)
        text += ch if ch else ""
    return "".join([c for c in text if c in string.ascii_uppercase])

# ---------------- State logic ----------------
def infer_tennis_state(set_cols, right_col, server_row):
    def is_completed_set(a,b):
        hi,lo=max(a,b),min(a,b)
        if hi<6: return False
        if hi>=6 and hi-lo>=2: return True
        if hi==7 and lo==6: return True
        return False
    sets_top=sets_bot=0; last_done=-1
    for i,(t,b) in enumerate(set_cols):
        if is_completed_set(t,b):
            last_done=i; sets_top += (t>b); sets_bot += (b>t)
        else: break
    cur_idx = last_done if last_done==len(set_cols)-1 else last_done+1
    is_tb=False; game={'type':'none','score':None}; ok=True; reason=''
    def map_pts(p):
        if p in (0,15,30,40): return str(p)
        if p in ('Ad','A'): return 'Ad'
        if isinstance(p,int): return ['0','15','30','40','Ad'][p] if 0<=p<=4 else str(p)
        return str(p)
    live=set_cols[cur_idx] if 0<=cur_idx<len(set_cols) else None
    if live and live[0]==6 and live[1]==6 and isinstance(right_col, tuple):
        a,b=right_col
        if all(isinstance(x,int) and x>=0 for x in (a,b)):
            is_tb=True; game={'type':'tiebreak','score':(a,b)}
        else:
            ok=False; reason='Expected integer tiebreak points at 6–6.'
    else:
        if right_col is None: game={'type':'points','score':None}
        elif isinstance(right_col, tuple): game={'type':'points','score':(map_pts(right_col[0]), map_pts(right_col[1]))}
        else: game={'type':'points','score':right_col}
    return {'currentSetIndex':cur_idx,'setsWonTop':sets_top,'setsWonBottom':sets_bot,
            'perSetScores':list(set_cols),'isTiebreak':is_tb,'gameScore':game,
            'server':('top' if server_row==0 else 'bottom') if server_row in (0,1) else 'unknown',
            'ok':ok,'reason':reason}

# ---------------- Pipeline ----------------
def run_pipeline(image_path: Path, result_json: Path=None, overlay_path: Path=None, segments_path: Path=None):
    im,g=imread_gray(image_path); H,W=g.shape[:2]

    # detect
    digit_boxes=nms_union(filter_digitish(nms_union(mser_boxes(g),0.3), W,H), 0.2)
    name_boxes,_=find_name_boxes(g)

    # server
    serve_markers, masks = detect_server_markers(im, name_boxes)
    server_row = None
    if serve_markers:
        serve_markers = sorted(serve_markers, key=lambda d:d['bbox'][0])
        server_row = serve_markers[0]['row']

    # names OCR
    pyt=try_import_tesseract()
    name_boxes_sorted=sorted(name_boxes, key=lambda b:b[1])
    names=[ocr_name_crop(im[y:y+h, x:x+w], pyt) for (x,y,w,h) in name_boxes_sorted]

    # digits → columns
    med_y=np.median([y+h/2 for (_,y,_,h) in digit_boxes]) if digit_boxes else H/2
    cols={}
    for (x,y,w,h) in digit_boxes:
        v=ocr_digit_crop(im[y:y+h, x:x+w], pyt)
        r=0 if (y+h/2)<=med_y else 1
        if v is None: continue
        xc=x+w/2; found=None
        for k in list(cols.keys()):
            if abs(k-xc)<W*0.02: found=k; break
        if found is None: cols[xc]=[None,None]; found=xc
        cols[found][r]=v
    sorted_cols=[tuple(cols[k]) for k in sorted(cols.keys())]
    set_cols=sorted_cols[:-1] if len(sorted_cols)>=3 else sorted_cols
    right_col=sorted_cols[-1] if len(sorted_cols)>=3 else None

    state=infer_tennis_state(set_cols, right_col, server_row)

    result={"image":str(image_path),
            "players":{"top":names[0] if len(names)>0 else "", "bottom":names[1] if len(names)>1 else ""},
            "setScores":set_cols,"rightColumn":right_col,
            "serverRow":server_row,"serveMarkers":serve_markers,"state":state}

    # overlay
    if overlay_path:
        vis=im.copy()
        for b in digit_boxes: rect(vis,b,(0,255,255),2)
        for b in name_boxes_sorted: rect(vis,b,(255,0,255),2)
        for d in serve_markers:
            rect(vis, d['bbox'], (0,255,0), 2)
            x,y,_,_=d['bbox']; cv2.putText(vis, f"S{d['row']}:{d['score']:.2f}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imwrite(str(overlay_path), vis)

    # segmentation sheet
    if segments_path and SAVE_DEBUG_SHEETS:
        color_m=put_label(masks['color_mask'], "color mask")
        bright_m=put_label(masks['bright_mask'], "bright mask")
        cand_m=put_label(masks['cand'], "combined mask")
        im_digits=put_label(draw_boxes(im, digit_boxes,(0,255,255),2), "digits")
        im_names=put_label(draw_boxes(im, name_boxes_sorted,(255,0,255),2), "names")
        im_srv=im.copy()
        for d in serve_markers: rect(im_srv, d['bbox'], (0,255,0), 2)
        im_srv=put_label(im_srv,"server marker(s)")
        sheet=make_sheet([put_label(im,"original"), im_digits, im_names],
                         [to_bgr(color_m), to_bgr(bright_m), to_bgr(cand_m)],
                         [im_srv])
        cv2.imwrite(str(segments_path), sheet)

    if result_json:
        with open(result_json,"w") as f: json.dump(result,f,indent=2)
    return result

# ---------------- Batch ----------------
def run_batch(paths, out_dir="scorebug_out"):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results=[]
    for p in paths:
        p=Path(p)
        overlay  = out_dir/(p.stem+"_overlay.png")
        segments = out_dir/(p.stem+"_segments.png")
        js       = out_dir/(p.stem+"_result.json")
        try:
            r=run_pipeline(p, js, overlay, segments)
        except Exception as e:
            r={"image":str(p), "error":str(e)}
        results.append(r)
    with open(out_dir/"batch_summary.json","w") as f:
        json.dump(results,f,indent=2)
    return results

def main():
    argv=sys.argv[1:] or ["data_scorebugs/"]  # default to your folder
    paths=[]
    for a in argv:
        if any(ch in a for ch in ["*","?","["]):
            paths.extend(glob.glob(a))
        else:
            p=Path(a)
            if p.is_dir(): paths.extend(glob.glob(str(p/"*.*")))
            else: paths.append(str(p))
    run_batch(paths)

if __name__=="__main__":
    main()
