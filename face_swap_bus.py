#!/usr/bin/env python3
"""
AuroStrike F1 Bus Face Swap
Swaps faces of bus passengers with F1 drivers + adds bulldog with sunglasses.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

BUS_IMAGE    = os.path.join(WORK_DIR, "bus.png")
AURORA_IMAGE = os.path.join(WORK_DIR, "aurora.jpg")
OUTPUT_IMAGE = os.path.join(WORK_DIR, "aurostrike_bus_final.jpg")
DRIVERS_DIR  = os.path.join(WORK_DIR, "drivers")
MODEL_PATH   = "/root/.insightface/models/inswapper_128.onnx"

# Mapping: seat position label -> driver image filename
# Positions are matched to detected faces sorted left-to-right, row by row
DRIVER_ASSIGNMENTS = {
    # index in sorted face list -> driver name
    0: "leclerc",     # Pos 1 – front left
    1: "verstappen",  # Pos 2 – front center (phone)
    2: "alonso",      # Pos 3 – front right (denim jacket)
    3: "piastri",     # Pos 4 – mid left (smirk)
    4: "hamilton",    # Pos 5 – mid right (big smile)
    5: "antonelli",   # Pos 6 – standing at back
    6: "aurora",      # Pos 7 – last person at back
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cv2(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def sort_faces(faces):
    """Sort faces top→bottom, left→right (row-major by y-band)."""
    if not faces:
        return faces
    # Use bbox centre y to group rows (tolerance = 80px)
    boxes = [(f.bbox[0], f.bbox[1]) for f in faces]
    avg_h = np.mean([f.bbox[3] - f.bbox[1] for f in faces])
    rows = []
    remaining = list(zip(range(len(faces)), faces))
    while remaining:
        top_y = min(y for _, f in remaining for y in [f.bbox[1]])
        row, rest = [], []
        for idx, f in remaining:
            if f.bbox[1] <= top_y + avg_h * 0.6:
                row.append((idx, f))
            else:
                rest.append((idx, f))
        row.sort(key=lambda x: x[1].bbox[0])
        rows.extend(row)
        remaining = rest
    return [f for _, f in rows]


# ---------------------------------------------------------------------------
# Face swap using insightface INSwapper
# ---------------------------------------------------------------------------

def run_face_swap():
    print("\n=== AuroStrike F1 Bus Face Swap ===\n")

    # --- Check required files ---
    missing = []
    if not os.path.exists(BUS_IMAGE):
        missing.append("bus.png")
    if not os.path.exists(AURORA_IMAGE):
        missing.append("aurora.jpg")
    if not os.path.exists(MODEL_PATH):
        missing.append(f"inswapper model at {MODEL_PATH}")
    if missing:
        print("ERROR – missing files:")
        for m in missing:
            print(f"  • {m}")
        sys.exit(1)

    # --- Load insightface ---
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    print("Loading face analysis model...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print("Loading INSwapper model...")
    swapper = get_model(MODEL_PATH)

    # --- Load bus image ---
    print(f"Loading bus image: {BUS_IMAGE}")
    bus_bgr = load_cv2(BUS_IMAGE)
    result   = bus_bgr.copy()

    # --- Detect faces in bus ---
    print("Detecting faces in bus image...")
    bus_faces = app.get(bus_bgr)
    if not bus_faces:
        print("ERROR: No faces detected in bus image!")
        sys.exit(1)

    sorted_bus_faces = sort_faces(bus_faces)
    print(f"  Found {len(sorted_bus_faces)} faces (sorted left→right, top→bottom)")

    # --- Process each assignment ---
    for pos_idx, driver_name in DRIVER_ASSIGNMENTS.items():
        if pos_idx >= len(sorted_bus_faces):
            print(f"  SKIP pos {pos_idx+1} – no face detected at that position")
            continue

        target_face = sorted_bus_faces[pos_idx]

        # Load driver source image
        if driver_name == "aurora":
            src_path = AURORA_IMAGE
        else:
            src_path = os.path.join(DRIVERS_DIR, f"{driver_name}.png")

        if not os.path.exists(src_path):
            print(f"  SKIP {driver_name} – image not found: {src_path}")
            continue

        src_bgr   = load_cv2(src_path)
        src_faces = app.get(src_bgr)
        if not src_faces:
            print(f"  SKIP {driver_name} – no face detected in source image")
            continue

        src_face = src_faces[0]  # use the first (most prominent) face

        print(f"  Swapping pos {pos_idx+1} → {driver_name}")
        result = swapper.get(result, target_face, src_face, paste_back=True)

    # --- Add bulldog with sunglasses ---
    result = add_bulldog(result)

    # --- Save ---
    cv2.imwrite(OUTPUT_IMAGE, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSaved: {OUTPUT_IMAGE}")
    return result


# ---------------------------------------------------------------------------
# Bulldog compositing (PIL)
# ---------------------------------------------------------------------------

def add_bulldog(bus_bgr):
    """
    Add a bulldog puppy with sunglasses on an empty/partial seat.
    The bulldog is composited using PIL with transparency handling.
    """
    from PIL import Image, ImageEnhance

    bus_pil = cv2_to_pil(bus_bgr)
    bw, bh  = bus_pil.size

    # Load bulldog (prefer no-background version from drivers dir)
    bulldog_path_nobg = os.path.join(DRIVERS_DIR, "bulldog_nobg.png")
    bulldog_path_std  = os.path.join(DRIVERS_DIR, "bulldog.png")

    bulldog = None

    # Try pre-downloaded no-bg version
    for path in [bulldog_path_nobg, bulldog_path_std]:
        if os.path.exists(path):
            bulldog = Image.open(path).convert("RGBA")
            print(f"  Bulldog loaded from: {path}")
            break

    if bulldog is None:
        # Generate a simple cartoon bulldog placeholder
        print("  Generating placeholder bulldog...")
        bulldog = generate_bulldog_placeholder()

    # --- Add sunglasses overlay ---
    bulldog = add_sunglasses_to_bulldog(bulldog)

    # --- Resize to ~seat-head size ---
    target_h = int(bh * 0.18)
    ratio     = target_h / bulldog.height
    target_w  = int(bulldog.width * ratio)
    bulldog   = bulldog.resize((target_w, target_h), Image.LANCZOS)

    # --- Position: left-side empty seat area (adjust as needed) ---
    # Place near the front-left area where there appears to be space
    paste_x = int(bw * 0.04)
    paste_y = int(bh * 0.45)

    # Ensure we stay within bounds
    paste_x = max(0, min(paste_x, bw - target_w))
    paste_y = max(0, min(paste_y, bh - target_h))

    # Slight shadow for realism
    shadow = Image.new("RGBA", (target_w + 6, target_h + 6), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 80))
    shadow.paste(shadow_layer, (4, 4))
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    bus_pil = bus_pil.convert("RGBA")
    bus_pil.paste(shadow, (paste_x - 2, paste_y - 2), shadow)

    # Paste bulldog
    bus_pil.paste(bulldog, (paste_x, paste_y), bulldog)
    bus_pil = bus_pil.convert("RGB")

    print(f"  Bulldog placed at ({paste_x}, {paste_y}) size {target_w}×{target_h}")
    return pil_to_cv2(bus_pil)


def add_sunglasses_to_bulldog(bulldog_rgba):
    """Overlay cartoon sunglasses on top portion of bulldog image."""
    w, h = bulldog_rgba.size
    draw  = ImageDraw.Draw(bulldog_rgba)

    # Sunglasses area: roughly top 40% of image, centered
    sg_y   = int(h * 0.15)
    sg_h   = int(h * 0.22)
    sg_w   = int(w * 0.75)
    sg_x   = (w - sg_w) // 2
    lens_w = (sg_w - 8) // 2
    lens_h = sg_h

    # Left lens
    draw.ellipse([sg_x, sg_y, sg_x + lens_w, sg_y + lens_h],
                 fill=(20, 20, 20, 220), outline=(10, 10, 10, 255), width=2)
    # Right lens
    draw.ellipse([sg_x + lens_w + 8, sg_y, sg_x + sg_w, sg_y + lens_h],
                 fill=(20, 20, 20, 220), outline=(10, 10, 10, 255), width=2)
    # Bridge
    mid = sg_x + lens_w
    draw.line([mid, sg_y + lens_h//2, mid + 8, sg_y + lens_h//2],
              fill=(80, 80, 80, 255), width=2)
    # Temple left
    draw.line([sg_x, sg_y + lens_h//2, sg_x - 6, sg_y + lens_h//2],
              fill=(80, 80, 80, 255), width=2)
    # Temple right
    draw.line([sg_x + sg_w, sg_y + lens_h//2, sg_x + sg_w + 6, sg_y + lens_h//2],
              fill=(80, 80, 80, 255), width=2)

    return bulldog_rgba


def generate_bulldog_placeholder():
    """Create a simple cartoon bulldog when no real image is available."""
    size = (200, 200)
    img  = Image.new("RGBA", size, (0, 0, 0, 0))
    d    = ImageDraw.Draw(img)
    w, h = size

    # Body
    d.ellipse([30, 90, 170, 190], fill=(180, 140, 100, 255))
    # Head
    d.ellipse([50, 30, 155, 120], fill=(200, 160, 110, 255))
    # Ears
    d.ellipse([35, 30, 75, 75], fill=(160, 120, 80, 255))
    d.ellipse([130, 30, 170, 75], fill=(160, 120, 80, 255))
    # Eyes
    d.ellipse([75, 55, 95, 75], fill=(255, 255, 255, 255))
    d.ellipse([110, 55, 130, 75], fill=(255, 255, 255, 255))
    d.ellipse([80, 60, 91, 71], fill=(50, 30, 10, 255))
    d.ellipse([115, 60, 126, 71], fill=(50, 30, 10, 255))
    # Nose
    d.ellipse([88, 85, 115, 100], fill=(80, 50, 40, 255))
    # Mouth
    d.arc([80, 95, 125, 115], 0, 180, fill=(80, 50, 40, 255), width=3)
    # Wrinkles
    d.line([85, 80, 80, 90], fill=(160, 120, 80, 255), width=2)
    d.line([118, 80, 123, 90], fill=(160, 120, 80, 255), width=2)

    return img


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_face_swap()
    print("\nDone! Output saved to:", OUTPUT_IMAGE)
