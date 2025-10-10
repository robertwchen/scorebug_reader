from PIL import Image
from pathlib import Path
import pytesseract

# 1) point pytesseract at your tesseract.exe (windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 2) pick a test image; put a real scorebug screenshot here
imgPath = Path("scorebug_test.png")  # e.g., export a single frame from a match

# 3) load the image (fail fast if missing)
if not imgPath.exists():
    raise FileNotFoundError(f"Image not found: {imgPath.resolve()}")

img = Image.open(imgPath)

# 4) do the simplest possible OCR
#    --oem 3: default LSTM engine
#    --psm 6: "Assume a block of text" (good default to start)
config = "--oem 3 --psm 6"
rawText = pytesseract.image_to_string(img, config=config)

print("=== OCR (raw) ===")
print(rawText)