# ocr_sweep_simple.py
# Goal: load one image -> optional ROI -> optional preprocessing -> try Tesseract layouts
# Read outputs and pick what works. Keep it simple.

from pathlib import Path
from PIL import Image
import pytesseract
from pytesseract import Output
import sys
import re

# ---------- edit these ----------
imagePath = "scorebug_test.png"   # put your frame here
useRoi = False                    # set True and tune roiFrac if you want a crop
roiFrac = (0.60, 0.05, 0.98, 0.18)  # (x0_frac, y0_frac, x1_frac, y1_frac) if useRoi=True
usePreprocess = True              # try grayscale -> 2x upscale -> threshold
useWhitelist = True               # restrict to digits and score symbols in early tests
showBoxes = False                 # print word boxes and confidences
expectedText = ""                 # optional, set like "2-6 6-1" for a quick quality number

# Windows: set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- tiny helpers ----------
def norm(s):
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def digitRatio(s):
    if not s:
        return 0.0
    count = 0
    for c in s:
        if c.isdigit():
            count += 1
    return count / len(s)

def levenshtein(a, b):
    # simple DP distance. returns 0..max(len(a),len(b))
    a = norm(a)
    b = norm(b)
    if not a and not b:
        return 0
    m = len(a)
    n = len(b)
    dp = [0] * (n + 1)
    j = 0
    while j <= n:
        dp[j] = j
        j += 1
    i = 1
    while i <= m:
        prev = dp[0]
        dp[0] = i
        j = 1
        while j <= n:
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            delCost = dp[j] + 1
            insCost = dp[j - 1] + 1
            subCost = prev + cost
            best = delCost
            if insCost < best:
                best = insCost
            if subCost < best:
                best = subCost
            dp[j] = best
            prev = temp
            j += 1
        i += 1
    return dp[n]

def normEditDistance(a, b):
    if not b:
        return None
    dist = levenshtein(a, b)
    denom = max(len(norm(a)), len(norm(b)))
    if denom == 0:
        return 0.0
    return dist / denom

def cropRoi(img, frac):
    W, H = img.size
    x0 = int(W * frac[0]); y0 = int(H * frac[1])
    x1 = int(W * frac[2]); y1 = int(H * frac[3])
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x1 > W: x1 = W
    if y1 > H: y1 = H
    return img.crop((x0, y0, x1, y1))

def makeVariants(img, doPrep):
    variants = []
    variants.append(("orig", img))
    if not doPrep:
        return variants

    gray = img.convert("L")
    variants.append(("gray", gray))

    up2 = gray.resize((gray.width * 2, gray.height * 2), Image.NEAREST)
    variants.append(("up2", up2))

    thr = 150
    bw = up2.point(lambda p: 255 if p > thr else 0)
    variants.append((f"bw{thr}", bw))
    return variants

def runConfig(img, cfg, expected, showBoxesFlag):
    text = pytesseract.image_to_string(img, config=cfg)
    clean = norm(text)
    dr = digitRatio(clean)
    ed = normEditDistance(clean, expected) if expected else None

    print("\n" + "=" * 70)
    print("CONFIG:", cfg)
    print("RAW:", repr(text))
    print("CLEAN:", clean)
    line = "len=" + str(len(clean)) + "  digit_ratio=" + "{:.2f}".format(dr)
    if ed is not None:
        line += "  norm_edit_dist_to_expected=" + "{:.2f}".format(ed)
    print(line)

    if showBoxesFlag:
        data = pytesseract.image_to_data(img, config=cfg, output_type=Output.DICT)
        total = len(data["text"])
        print("BOXES (conf >= 0):")
        i = 0
        while i < total:
            t = data["text"][i].strip()
            confStr = data["conf"][i]
            try:
                conf = int(float(confStr))
            except:
                conf = -1
            if t and conf >= 0:
                x = data["left"][i]; y = data["top"][i]
                w = data["width"][i]; h = data["height"][i]
                print("  [" + str(conf).rjust(3) + "] (" + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + ") -> " + t)
            i += 1


# ---------- main flow ----------
def main():
    imgPath = Path(imagePath)
    if not imgPath.exists():
        print("Image not found:", imgPath.resolve(), file=sys.stderr)
        sys.exit(1)

    print("Tesseract version:", pytesseract.get_tesseract_version(), file=sys.stderr)

    img = Image.open(imgPath)
    print("Image mode/size:", img.mode, img.size, file=sys.stderr)
    img.save("preview_original.png")

    if useRoi:
        img = cropRoi(img, roiFrac)
        img.save("preview_roi.png")

    variants = makeVariants(img, usePreprocess)

    # configs to try (small, focused set for scorebugs)
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 7",
        "--oem 3 --psm 8",
        "--oem 3 --psm 11",
        "--oem 3 --psm 13",
        "--oem 1 --psm 7",
        "--oem 1 --psm 8",
        "--oem 1 --psm 11",
        "--oem 1 --psm 13",
    ]
    if useWhitelist:
        wl = " -c tessedit_char_whitelist=0123456789-:()"
    else:
        wl = ""

    # sweep: for each image variant, try each layout config
    for name, pic in variants:
        print("\n" + "#" * 70)
        print("VARIANT:", name)
        for cfg in configs:
            fullCfg = cfg + wl
            runConfig(pic, fullCfg, expectedText, showBoxes)

if __name__ == "__main__":
    main()
