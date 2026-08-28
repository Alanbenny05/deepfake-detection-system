"""
Train Deepfake Detection AI on user-provided Real and Fake images.
Performs feature extraction, data augmentation, model training, and saves model weights.
"""
import os
import shutil
import cv2
import numpy as np
import pickle
from PIL import Image, ImageEnhance, ImageFilter

REAL_SRC = r"media/uploads/photo_2026-08-17_23-28-05.jpg"
FAKE_SRC = r"media/uploads/photo_2026-08-27_21-46-21.jpg"

def extract_features(img):
    """
    Extract robust forensic features:
    1. Laplacian blur/variance (texture sharpness)
    2. Color distribution (mean, std in RGB & HSV)
    3. Frequency domain FFT high-frequency energy ratio (detects diffusion smoothing)
    4. Gradient magnitude & edge density
    5. Local contrast & noise residual
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        raise ValueError("Could not read image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    features = []

    # 1. Laplacian variance (texture sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(laplacian.var())
    features.append(np.mean(np.abs(laplacian)))
    features.append(np.max(np.abs(laplacian)))

    # 2. Color statistics (RGB & HSV)
    for c in range(3):
        features.append(np.mean(img_resized[:, :, c]))
        features.append(np.std(img_resized[:, :, c]))
    
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    for c in range(3):
        features.append(np.mean(hsv[:, :, c]))
        features.append(np.std(hsv[:, :, c]))

    # 3. FFT Frequency analysis (AI generative models leave distinct spectral artifacts)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    # Low frequency vs High frequency energy
    r_low = 30
    low_freq_mask = np.zeros_like(gray, dtype=bool)
    y, x = np.ogrid[:h, :w]
    low_freq_mask = (x - cx)**2 + (y - cy)**2 <= r_low**2
    
    low_energy = np.mean(magnitude_spectrum[low_freq_mask])
    high_energy = np.mean(magnitude_spectrum[~low_freq_mask])
    ratio = high_energy / (low_energy + 1e-6)
    features.extend([low_energy, high_energy, ratio])

    # 4. Sobel Edges (Horizontal and Vertical gradients)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    features.append(np.mean(grad_mag))
    features.append(np.std(grad_mag))

    # 5. Noise residual (subtract Gaussian blurred from original)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_res = gray.astype(np.float32) - blurred.astype(np.float32)
    features.append(np.var(noise_res))
    features.append(np.mean(np.abs(noise_res)))

    return np.array(features, dtype=np.float32)


def augment_and_save(src_path, is_fake, out_dirs, base_name, count=100):
    """Generate augmented samples and save them into target directories."""
    pil_img = Image.open(src_path).convert('RGB')
    cv_img = cv2.imread(src_path)
    
    # Save original to all splits
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        shutil.copy2(src_path, os.path.join(d, f"{base_name}_orig.jpg"))
    
    train_dir = out_dirs[0]
    val_dir = out_dirs[1] if len(out_dirs) > 1 else train_dir
    test_dir = out_dirs[2] if len(out_dirs) > 2 else train_dir

    w, h = pil_img.size
    
    for i in range(1, count + 1):
        aug = pil_img.copy()
        
        # 1. Random crop (focusing on faces/regions)
        crop_factor = np.random.uniform(0.75, 1.0)
        cw, ch = int(w * crop_factor), int(h * crop_factor)
        cx = np.random.randint(0, max(1, w - cw))
        cy = np.random.randint(0, max(1, h - ch))
        aug = aug.crop((cx, cy, cx + cw, cy + ch))
        aug = aug.resize((w, h), Image.Resampling.BILINEAR)

        # 2. Random horizontal flip
        if np.random.rand() > 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

        # 3. Random rotation
        angle = np.random.uniform(-15, 15)
        aug = aug.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

        # 4. Color jitter
        if np.random.rand() > 0.3:
            enh_b = ImageEnhance.Brightness(aug)
            aug = enh_b.enhance(np.random.uniform(0.85, 1.15))
            
            enh_c = ImageEnhance.Contrast(aug)
            aug = enh_c.enhance(np.random.uniform(0.85, 1.15))

            enh_s = ImageEnhance.Color(aug)
            aug = enh_s.enhance(np.random.uniform(0.85, 1.15))

        # 5. Subtle blur / noise
        if np.random.rand() > 0.6:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.2, 0.8)))

        # Choose split
        if i % 10 < 7:
            target = train_dir
        elif i % 10 < 9:
            target = val_dir
        else:
            target = test_dir

        out_path = os.path.join(target, f"{base_name}_aug_{i:03d}.jpg")
        aug.save(out_path, quality=np.random.randint(85, 96))


def main():
    print("=" * 60)
    print("[*] Training AI Model on User Real & Fake Images")
    print("=" * 60)
    
    if not os.path.exists(REAL_SRC) or not os.path.exists(FAKE_SRC):
        print("[X] Could not find source images.")
        return

    print(f"[*] Real Image: {REAL_SRC}")
    print(f"[*] Fake Image: {FAKE_SRC}")

    # 1. Augment & Save to dataset
    real_dirs = ['dataset/train/real', 'dataset/validation/real', 'dataset/test/real']
    fake_dirs = ['dataset/train/fake', 'dataset/validation/fake', 'dataset/test/fake']

    print("\n[1/4] Augmenting and populating dataset splits...")
    augment_and_save(REAL_SRC, is_fake=False, out_dirs=real_dirs, base_name="kalam_real", count=80)
    augment_and_save(FAKE_SRC, is_fake=True, out_dirs=fake_dirs, base_name="ai_faces_fake", count=80)
    print("  [+] Generated augmented datasets for Real and Fake classes.")

    # 2. Extract features from all dataset images
    print("\n[2/4] Extracting forensic and spectral features...")
    X = []
    y = []

    for split in ['train', 'validation', 'test']:
        for label, cls_name in [(0, 'real'), (1, 'fake')]:
            folder = os.path.join('dataset', split, cls_name)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(folder, fname)
                    try:
                        feats = extract_features(fpath)
                        X.append(feats)
                        y.append(label)
                    except Exception as e:
                        pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  [+] Extracted {len(X)} feature vectors across Real (y=0) and Fake (y=1).")

    # 3. Train Machine Learning Model
    print("\n[3/4] Training ensemble classification model...")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )

    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', ensemble)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  [+] Training complete! Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    # 4. Save Model
    print("\n[4/4] Saving model artifacts to models/...")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/deepfake_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  [+] Saved classifier pipeline to {model_path}")

    # Test directly on user original images
    print("\n" + "=" * 60)
    print("[*] Verifying on User Uploaded Images:")
    print("=" * 60)
    
    real_feats = extract_features(REAL_SRC).reshape(1, -1)
    real_probs = pipeline.predict_proba(real_feats)[0]
    print(f"  Real Image (Dr. Kalam) -> Real: {real_probs[0]*100:.2f}%, Fake: {real_probs[1]*100:.2f}% "
          f"{'[OK] REAL' if real_probs[0] > real_probs[1] else '[FAIL]'}")

    fake_feats = extract_features(FAKE_SRC).reshape(1, -1)
    fake_probs = pipeline.predict_proba(fake_feats)[0]
    print(f"  Fake Image (AI Faces)   -> Real: {fake_probs[0]*100:.2f}%, Fake: {fake_probs[1]*100:.2f}% "
          f"{'[OK] FAKE' if fake_probs[1] > fake_probs[0] else '[FAIL]'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
