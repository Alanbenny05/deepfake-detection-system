"""
Generate a large, realistic synthetic dataset for deepfake detection training.

REAL images: natural skin tones, soft bokeh blur, realistic camera noise (Gaussian + JPEG artefacts)
FAKE images: GAN-style artefacts — checkerboard noise, blending halo, DCT frequency anomalies,
             colour desaturation, pixel-level noise patterns.

Volumes: 500 train / 100 validation / 100 test images per class.
"""
import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io

# ─────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────

def _rng(seed):
    rng = random.Random(seed)
    return rng


def _add_gaussian_noise(img_array, std=6.0, rng=None):
    """Add realistic camera sensor noise."""
    if rng is None:
        rng = random.Random()
    noise = np.random.RandomState(rng.randint(0, 2**31)).randn(*img_array.shape) * std
    noisy = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def _jpeg_roundtrip(img, quality=None, rng=None):
    """Simulate JPEG compression artefacts (realistic camera/sharing behaviour)."""
    if rng is None:
        rng = random.Random()
    if quality is None:
        quality = rng.randint(75, 95)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _add_checkerboard_noise(img_array, strength=18, block=4, rng=None):
    """
    GAN artefact: high-frequency checkerboard pattern from transposed convolutions.
    Visible in DCT spectrum of GAN-generated images.
    """
    if rng is None:
        rng = random.Random()
    h, w, c = img_array.shape
    pattern = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if (y // block + x // block) % 2 == 0:
                pattern[y:y+block, x:x+block] = strength
            else:
                pattern[y:y+block, x:x+block] = -strength
    # Apply only to specific channels to simulate colour channel artefacts
    ch = rng.randint(0, c - 1)
    result = img_array.astype(np.float32).copy()
    result[:, :, ch] = np.clip(result[:, :, ch] + pattern, 0, 255)
    return result.astype(np.uint8)


def _add_blending_halo(img, face_bbox, strength=0.35, rng=None):
    """
    GAN artefact: unnatural blending boundary around the face region.
    Deepfakes often have a subtle halo / soft edge around pasted face regions.
    """
    if rng is None:
        rng = random.Random()
    overlay = img.copy().convert('RGBA')
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = face_bbox
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    rx = (x1 - x0) // 2 + rng.randint(2, 8)
    ry = (y1 - y0) // 2 + rng.randint(2, 8)
    halo_color = (
        rng.randint(180, 255),
        rng.randint(180, 255),
        rng.randint(180, 255),
        int(strength * 255)
    )
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=halo_color, width=rng.randint(2, 5))
    result = Image.alpha_composite(overlay, Image.new('RGBA', img.size, (0, 0, 0, 0)))
    return result.convert('RGB')


def _add_frequency_artefacts(img_array, rng=None):
    """
    Simulate DCT-domain high-frequency noise typical of GAN generators.
    Adds subtle periodic patterns that are invisible to the eye but captured by the model.
    """
    if rng is None:
        rng = random.Random()
    h, w, c = img_array.shape
    result = img_array.astype(np.float32).copy()
    freq_x = rng.choice([4, 8, 16])
    freq_y = rng.choice([4, 8, 16])
    amp = rng.uniform(4.0, 12.0)
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    wave = amp * np.sin(2 * math.pi * xv / freq_x) * np.cos(2 * math.pi * yv / freq_y)
    ch = rng.randint(0, c - 1)
    result[:, :, ch] = np.clip(result[:, :, ch] + wave, 0, 255)
    return result.astype(np.uint8)


def _desaturate_region(img, face_bbox, amount=0.25):
    """Partial desaturation of face region — a known GAN colour artefact."""
    x0, y0, x1, y1 = face_bbox
    region = img.crop((x0, y0, x1, y1))
    enhancer = ImageEnhance.Color(region)
    desaturated = enhancer.enhance(1.0 - amount)
    result = img.copy()
    result.paste(desaturated, (x0, y0))
    return result


# ─────────────────────────────────────────────────────────────
#  Face image generators
# ─────────────────────────────────────────────────────────────

def _draw_base_face(size, rng):
    """Draw a realistic-looking synthetic face on a PIL Image."""
    w, h = size
    # Background — varied indoor/outdoor tones
    bg_palette = [
        (210, 210, 215),  # grey indoor
        (195, 215, 230),  # blue-ish sky
        (220, 225, 200),  # outdoor green
        (235, 225, 215),  # warm indoor
        (180, 190, 205),  # cool shadow
    ]
    bg_color = rng.choice(bg_palette)
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Subtle background texture — soft ellipses
    for _ in range(20):
        bx = rng.randint(-20, w + 20)
        by = rng.randint(-20, h + 20)
        br = rng.randint(8, 35)
        bc = tuple(max(0, min(255, bg_color[i] + rng.randint(-15, 15))) for i in range(3))
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=bc)

    # Skin tones — diverse range
    skin_palettes = [
        (240, 210, 185),   # fair
        (225, 190, 160),   # light
        (200, 165, 130),   # medium
        (175, 135, 100),   # tan
        (145, 105, 75),    # brown
        (115, 80, 55),     # dark
        (90, 60, 40),      # deep
    ]
    skin = rng.choice(skin_palettes)
    # Slight variation within chosen skin tone
    skin = tuple(max(0, min(255, skin[i] + rng.randint(-10, 10))) for i in range(3))

    # Face oval — varied proportions
    face_scale_x = rng.uniform(0.52, 0.62)
    face_scale_y = rng.uniform(0.60, 0.72)
    fx0 = int(w * (0.5 - face_scale_x / 2))
    fy0 = int(h * (0.10 + rng.uniform(0, 0.06)))
    fx1 = int(w * (0.5 + face_scale_x / 2))
    fy1 = int(h * (0.10 + rng.uniform(0, 0.06)) + h * face_scale_y)
    face_bbox = (fx0, fy0, fx1, fy1)
    outline_color = tuple(max(0, c - 30) for c in skin)
    draw.ellipse(face_bbox, fill=skin, outline=outline_color, width=2)

    # Neck
    neck_w = int((fx1 - fx0) * 0.32)
    neck_top = int(fy1 - (fy1 - fy0) * 0.08)
    neck_x0 = w // 2 - neck_w // 2
    neck_x1 = w // 2 + neck_w // 2
    draw.rectangle([neck_x0, neck_top, neck_x1, min(h, fy1 + 30)], fill=skin)

    # Eyes
    face_cx = (fx0 + fx1) // 2
    face_cy = (fy0 + fy1) // 2
    eye_y = fy0 + int((fy1 - fy0) * 0.40)
    eye_sep = int((fx1 - fx0) * 0.22)
    eye_w = int((fx1 - fx0) * 0.14)
    eye_h = int((fy1 - fy0) * 0.075)

    for ex in [face_cx - eye_sep, face_cx + eye_sep]:
        # White of eye
        draw.ellipse([ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h],
                     fill=(255, 255, 255), outline=(60, 50, 45), width=1)
        # Iris colour — varied
        iris_colors = [(60, 35, 15), (50, 80, 50), (40, 70, 110), (70, 55, 30)]
        iris_color = rng.choice(iris_colors)
        ir = int(eye_w * 0.55)
        draw.ellipse([ex - ir, eye_y - ir, ex + ir, eye_y + ir], fill=iris_color)
        # Pupil
        pr = int(ir * 0.45)
        draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr], fill=(10, 10, 10))
        # Highlight
        draw.ellipse([ex - pr + 2, eye_y - pr + 2, ex - pr + 5, eye_y - pr + 5], fill=(240, 240, 255))

    # Eyebrows
    brow_y = eye_y - int(eye_h * 2.5)
    brow_thickness = rng.randint(2, 4)
    brow_color = tuple(max(0, c - 80) for c in skin)
    for ex in [face_cx - eye_sep, face_cx + eye_sep]:
        draw.line([(ex - eye_w, brow_y), (ex + eye_w, brow_y - rng.randint(1, 4))],
                  fill=brow_color, width=brow_thickness)

    # Nose
    nose_x = face_cx
    nose_y = fy0 + int((fy1 - fy0) * 0.57)
    nose_tip_w = int((fx1 - fx0) * 0.09)
    nose_h = int((fy1 - fy0) * 0.16)
    nose_color = tuple(max(0, c - 15) for c in skin)
    # Bridge
    draw.line([(nose_x, nose_y - nose_h), (nose_x, nose_y)], fill=nose_color, width=2)
    # Nostrils
    draw.ellipse([nose_x - nose_tip_w - 3, nose_y - 4,
                  nose_x - nose_tip_w + 5, nose_y + 5], fill=nose_color)
    draw.ellipse([nose_x + nose_tip_w - 5, nose_y - 4,
                  nose_x + nose_tip_w + 3, nose_y + 5], fill=nose_color)

    # Mouth
    mouth_y = fy0 + int((fy1 - fy0) * 0.75)
    mouth_w = int((fx1 - fx0) * 0.25)
    lip_colors = [
        (190, 80, 70), (210, 100, 90), (170, 60, 55),
        (200, 130, 115), (160, 50, 50),
    ]
    lip_color = rng.choice(lip_colors)
    # Upper lip arc
    draw.arc([face_cx - mouth_w, mouth_y - 8, face_cx + mouth_w, mouth_y + 4],
             start=200, end=340, fill=(max(0, lip_color[0] - 30), lip_color[1] - 15, lip_color[2] - 15), width=2)
    # Lower lip arc
    draw.arc([face_cx - mouth_w, mouth_y - 4, face_cx + mouth_w, mouth_y + 12],
             start=0, end=180, fill=lip_color, width=3)

    # Hair — simple block on top
    hair_colors = [(30, 20, 10), (60, 40, 25), (15, 10, 5), (150, 100, 55), (200, 170, 130)]
    hair_c = rng.choice(hair_colors)
    hair_h = int((fy1 - fy0) * 0.22)
    draw.ellipse([fx0 - 5, fy0 - hair_h, fx1 + 5, fy0 + hair_h], fill=hair_c)

    return img, face_bbox


def create_real_face_image(size=(224, 224), seed=None):
    """
    Generate a realistic 'real' face image with natural photographic characteristics.
    - Natural skin, varied lighting, subtle sensor noise, JPEG artefacts.
    """
    rng = _rng(seed)
    img, face_bbox = _draw_base_face(size, rng)

    # Subtle vignette (darkening edges) — realistic camera lens effect
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = 1 - np.clip(dist * 0.4, 0, 0.45)
    for c in range(3):
        arr[:, :, c] *= vignette
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Gaussian noise — camera sensor noise
    arr = _add_gaussian_noise(arr, std=rng.uniform(3.0, 8.0), rng=rng)

    # Slight Gaussian blur — lens focus / depth of field
    img = Image.fromarray(arr)
    blur_radius = rng.uniform(0.3, 1.0)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Random brightness/contrast tweak — lighting variation
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(rng.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(rng.uniform(0.90, 1.10))

    # JPEG roundtrip — typical camera/sharing compression
    img = _jpeg_roundtrip(img, quality=rng.randint(80, 96), rng=rng)

    return img


def create_fake_face_image(size=(224, 224), seed=None):
    """
    Generate a 'fake' (GAN-style deepfake) face image with common deepfake artefacts:
    - Checkerboard noise from transposed convolutions
    - Blending halo around face region
    - DCT frequency domain artefacts
    - Partial desaturation of face region
    - Colour channel inconsistencies
    """
    rng = _rng(seed)
    img, face_bbox = _draw_base_face(size, rng)
    arr = np.array(img)

    # Apply a random combination of deepfake artefacts
    artefacts = [True, True, True, True]  # checkerboard, halo, frequency, desaturation
    # Vary intensity per sample
    if rng.random() > 0.3:
        arr = _add_checkerboard_noise(arr, strength=rng.randint(10, 25),
                                      block=rng.choice([4, 8]), rng=rng)

    if rng.random() > 0.2:
        arr = _add_frequency_artefacts(arr, rng=rng)

    img = Image.fromarray(arr)

    if rng.random() > 0.3:
        img = _add_blending_halo(img, face_bbox, strength=rng.uniform(0.2, 0.45), rng=rng)

    if rng.random() > 0.35:
        # Desaturate face region by 20-40%
        img = _desaturate_region(img, face_bbox, amount=rng.uniform(0.2, 0.40))

    # Unnatural sharpness / edge enhancement — GAN over-sharpening
    if rng.random() > 0.4:
        img = img.filter(ImageFilter.EDGE_ENHANCE)

    # Colour channel saturation boost — GAN colour shift
    if rng.random() > 0.4:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(rng.uniform(1.15, 1.45))

    # Add subtle periodic pixel noise across the whole image
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(rng.randint(0, 2**31)).randn(*arr.shape) * rng.uniform(5.0, 14.0)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # JPEG at slightly lower quality — typical deepfake pipeline re-encoding
    img = _jpeg_roundtrip(img, quality=rng.randint(70, 88), rng=rng)

    return img


# ─────────────────────────────────────────────────────────────
#  Dataset generator
# ─────────────────────────────────────────────────────────────

def generate_dataset(
    base_dir='dataset',
    samples_per_category=None,
    overwrite_existing=False
):
    """
    Create directory structure and generate synthetic samples.

    Args:
        base_dir: Root dataset directory.
        samples_per_category: dict with keys 'train', 'validation', 'test' and int values.
        overwrite_existing: If False, skip folders that already have enough images.
    """
    if samples_per_category is None:
        samples_per_category = {'train': 500, 'validation': 100, 'test': 100}

    print(f"[*] Setting up dataset in '{base_dir}' ...")
    print(f"    Target: {samples_per_category} per class")

    splits = ['train', 'validation', 'test']
    classes = ['real', 'fake']
    total_generated = 0

    for split in splits:
        count = samples_per_category.get(split, 100)
        for cls in classes:
            target_dir = os.path.join(base_dir, split, cls)
            os.makedirs(target_dir, exist_ok=True)

            # Count existing images
            existing = [f for f in os.listdir(target_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not overwrite_existing and len(existing) >= count:
                print(f"  [skip] {split}/{cls}: already has {len(existing)} images")
                continue

            is_fake = (cls == 'fake')
            creator = create_fake_face_image if is_fake else create_real_face_image
            print(f"  Generating {count} {cls} images for {split}...", end='', flush=True)

            for i in range(1, count + 1):
                seed = abs(hash((split, cls, i))) % (2**31)
                img = creator(size=(224, 224), seed=seed)
                file_name = f"{cls}_{split}_{i:04d}.jpg"
                file_path = os.path.join(target_dir, file_name)
                img.save(file_path, format='JPEG', quality=92)
                total_generated += 1
                if i % 100 == 0:
                    print(f" {i}", end='', flush=True)

            print(f" ✓ ({count})")

    print(f"\n[+] Generated {total_generated} images in {base_dir}/")
    print("[+] Dataset ready for training.\n")
    return total_generated


if __name__ == '__main__':
    generate_dataset()
