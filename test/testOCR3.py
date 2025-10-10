# name_sweep.py
# Sweep OCR just for player names on your scorebug image (robust to dark bar + thin first glyphs)

from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageChops
import pytesseract, re, sys
from pytesseract import Output

# ---- inputs ----
imgPath = Path("scorebug_test.png")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ROIs (fractions). Tuned to avoid flag/green dot and the dark lower strip.
roiTop = (0.15, 0.08, 0.60, 0.48)   # WILLIAMS line
roiBot = (0.15, 0.52, 0.60, 0.88)   # SHARAPOVA line

# ---- helpers ----
def crop_frac(im, f):
    W, H = im.size
    x0, y0, x1, y1 = [int(W*f[0]), int(H*f[1]), int(W*f[2]), int(H*f[3])]
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
    return im.crop((x0, y0, x1, y1))

def clean(s): 
    return re.sub(r"\s+", " ", s.strip())

def alpha_ratio(s):
    if not s: return 0.0
    letters = sum(1 for c in s if c.isalpha())
    return letters/len(s)

def score(s):  # prefer more letters, then more characters
    return (alpha_ratio(s), len(s))

def ocr(im, cfg):
    return clean(pytesseract.image_to_string(im, config=cfg))

def boxes(im, cfg):
    d = pytesseract.image_to_data(im, config=cfg, output_type=Output.DICT)
    items = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t: 
            continue
        try:
            conf = int(float(d["conf"][i]))
        except:
            conf = -1
        if conf >= 0:
            items.append((conf, t, d["left"][i], d["top"][i], d["width"][i], d["height"][i]))
    return items

# ---------- name-friendly preprocessing variants ----------
def preprocess_variants(tile, scale=4):
    """
    Keep names in grayscale; avoid harsh binarization that floods dark bars.
    We try:
      - upscaled gray
      - autocontrast
      - mild sharpen
      - stroke thicken (MaxFilter) after autocontrast (helps W/S not split)
      - background flattening (subtract blurred background), then sharpen
      - optional mild binary at 160 (last resort)
    """
    v = []
    g = tile.convert("L")
    up = g.resize((g.width*scale, g.height*scale), Image.LANCZOS)

    up_auto = ImageOps.autocontrast(up, cutoff=1)
    up_unsharp = up_auto.filter(ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=2))
    up_thick = up_unsharp.filter(ImageFilter.MaxFilter(size=3))  # connect split strokes

    # background flatten: remove gradients so letters pop without flooding
    bg = up.filter(ImageFilter.BoxBlur(3))
    flat = ImageChops.subtract(up, bg)
    flat_auto = ImageOps.autocontrast(flat, cutoff=1)
    flat_unsharp = flat_auto.filter(ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=2))

    # very mild binary as a last option
    bw160 = up_auto.point(lambda p: 255 if p > 160 else 0)

    v.append(("up4_gray", up))
    v.append(("up4_auto", up_auto))
    v.append(("up4_unsharp", up_unsharp))
    v.append(("up4_thick", up_thick))
    v.append(("up4_flat", flat_unsharp))
    v.append(("up4_bw160", bw160))
    return v

def build_configs():
    base = [
        "--oem 3 --psm 7 --dpi 300",
        "--oem 3 --psm 13 --dpi 300",
        "--oem 1 --psm 7 --dpi 300",
        "--oem 1 --psm 13 --dpi 300",
    ]
    with_wl = [c + " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ " for c in base]
    no_wl = base[:]  # fallback without whitelist
    return with_wl + no_wl

def sweep_name(tile, who):
    best = {"score":(-1,-1), "text":"", "cfg":"", "var":"", "img":None, "boxes":[]}
    for vname, vimg in preprocess_variants(tile):
        for cfg in build_configs():
            t = ocr(vimg, cfg)
            sc = score(t)
            print(f"{who} | {vname:10s} | {cfg:35s} | CLEAN='{t}' | score={sc}")
            if sc > best["score"]:
                best.update({"score":sc, "text":t, "cfg":cfg, "var":vname, "img":vimg})
    best["boxes"] = boxes(best["img"], best["cfg"])
    return best

def draw_debug(full, rois, out="names_debug.png"):
    dbg = full.copy(); d = ImageDraw.Draw(dbg)
    for r in rois:
        W,H = full.size
        x0,y0,x1,y1 = int(W*r[0]),int(H*r[1]),int(W*r[2]),int(H*r[3])
        d.rectangle([x0,y0,x1,y1], outline="red", width=2)
    dbg.save(out)

# ---- main ----
def main():
    if not imgPath.exists():
        print("Image not found:", imgPath.resolve(), file=sys.stderr); sys.exit(1)
    full = Image.open(imgPath)

    topTile = crop_frac(full, roiTop)
    botTile = crop_frac(full, roiBot)
    draw_debug(full, [roiTop, roiBot])

    print("\n--- SWEEP: TOP NAME ---")
    bestTop = sweep_name(topTile, "TOP")

    print("\n--- SWEEP: BOTTOM NAME ---")
    bestBot = sweep_name(botTile, "BOT")

    # save the winning input bitmaps to inspect
    bestTop["img"].save("best_top_name_input.png")
    bestBot["img"].save("best_bot_name_input.png")

    print("\n=== RESULTS ===")
    print("TOP :", bestTop["text"], "| via", bestTop["var"], "|", bestTop["cfg"])
    print("BOT :", bestBot["text"], "| via", bestBot["var"], "|", bestBot["cfg"])

    print("\nTOP boxes:")
    for conf,t,x,y,w,h in bestTop["boxes"]:
        print(f"  [{conf:3}] ({x},{y},{w},{h}) -> {t}")
    print("\nBOT boxes:")
    for conf,t,x,y,w,h in bestBot["boxes"]:
        print(f"  [{conf:3}] ({x},{y},{w},{h}) -> {t}")

if __name__ == "__main__":
    main()
