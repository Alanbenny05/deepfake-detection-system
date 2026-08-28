"""
retrain_improved.py — One-Shot Deepfake Detection Retraining Script
=====================================================================

This script:
  1. Expands the synthetic dataset (500 train / 100 val / 100 test per class)
     with realistic GAN artefacts for fake images and natural noise for real.
  2. Trains the improved EfficientNetB0-based model in two stages.
  3. Saves the final model as models/deepfake_model.h5 (and backup paths).
  4. Evaluates on the test set and prints a summary.

Usage:
    python retrain_improved.py

Optional flags (edit at the top of this file):
    SKIP_DATASET_GENERATION  — set True if you already have a good dataset
    TRAIN_EPOCHS_STAGE1      — epochs for head-only training (default 25)
    TRAIN_EPOCHS_STAGE2      — epochs for fine-tuning (default 20)
"""

import os
import sys
import time

# ─────────────────────────────────────────────────────────────
#  Configuration — edit these as needed
# ─────────────────────────────────────────────────────────────

SKIP_DATASET_GENERATION = False   # Set True to reuse existing dataset

DATASET_SAMPLES = {
    'train':      500,   # per class
    'validation': 100,   # per class
    'test':       100,   # per class
}

TRAIN_EPOCHS_STAGE1 = 25
TRAIN_EPOCHS_STAGE2 = 20
UNFREEZE_LAYERS     = 30   # top N EfficientNetB0 layers to fine-tune


# ─────────────────────────────────────────────────────────────
#  Helper: pretty section header
# ─────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────
#  Step 1: Dataset
# ─────────────────────────────────────────────────────────────

def step_dataset():
    section("STEP 1 — Dataset Generation")
    from generate_sample_dataset import generate_dataset

    if SKIP_DATASET_GENERATION:
        print("[skip] SKIP_DATASET_GENERATION=True — using existing dataset.")
        return

    t0 = time.time()
    total = generate_dataset(base_dir='dataset', samples_per_category=DATASET_SAMPLES,
                             overwrite_existing=False)
    elapsed = time.time() - t0
    print(f"[done] {total} images ready in {elapsed:.1f}s")

    # Quick sanity check
    for split in ['train', 'validation', 'test']:
        for cls in ['real', 'fake']:
            p = os.path.join('dataset', split, cls)
            n = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}/{cls}: {n} images")


# ─────────────────────────────────────────────────────────────
#  Step 2: Train
# ─────────────────────────────────────────────────────────────

def step_train():
    section("STEP 2 — Model Training")

    # Verify TensorFlow is importable
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs available: {len(gpus)} - {[g.name for g in gpus] or 'CPU only'}")
    except ImportError:
        print("ERROR: TensorFlow is not installed.")
        print("Install it with:  pip install tensorflow  or  pip install tensorflow-cpu")
        sys.exit(1)

    from train_model import DeepfakeTrainer

    trainer = DeepfakeTrainer()
    trainer.create_model()
    trainer.prepare_data(data_dir='dataset')

    t0 = time.time()
    trainer.train_stage1(epochs=TRAIN_EPOCHS_STAGE1)
    trainer.train_stage2(epochs=TRAIN_EPOCHS_STAGE2, unfreeze_layers=UNFREEZE_LAYERS)
    elapsed = time.time() - t0

    print(f"\n[done] Training completed in {elapsed / 60:.1f} minutes")

    trainer.save_model()
    trainer.plot_training_history()

    return trainer


# ─────────────────────────────────────────────────────────────
#  Step 3: Evaluate
# ─────────────────────────────────────────────────────────────

def step_evaluate(trainer):
    section("STEP 3 — Evaluation on Test Set")
    trainer.evaluate_model(test_dir='dataset/test')


# ─────────────────────────────────────────────────────────────
#  Step 4: Quick smoke test using the new ml_model.py
# ─────────────────────────────────────────────────────────────

def step_smoke_test():
    section("STEP 4 — Smoke Test (new detector)")

    # Reset any cached detector instance
    import importlib
    import detector.ml_model as ml_module
    importlib.reload(ml_module)
    det = ml_module.DeepfakeDetector()

    results_summary = {'real_correct': 0, 'real_total': 0,
                       'fake_correct': 0, 'fake_total': 0}

    for cls in ['real', 'fake']:
        test_dir = os.path.join('dataset', 'test', cls)
        files = [f for f in os.listdir(test_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:20]  # first 20
        for fname in files:
            fpath = os.path.join(test_dir, fname)
            res = det.predict_image(fpath)
            results_summary[f'{cls}_total'] += 1
            if res['result'] == cls:
                results_summary[f'{cls}_correct'] += 1

    rt = results_summary['real_total']
    rc = results_summary['real_correct']
    ft = results_summary['fake_total']
    fc = results_summary['fake_correct']

    print(f"\n  Real images  : {rc}/{rt} correctly identified as real "
          f"({'N/A' if rt == 0 else f'{rc/rt*100:.0f}%'})")
    print(f"  Fake images  : {fc}/{ft} correctly identified as fake "
          f"({'N/A' if ft == 0 else f'{fc/ft*100:.0f}%'})")
    print(f"  Overall      : {rc+fc}/{rt+ft} correct "
          f"({'N/A' if rt+ft == 0 else f'{(rc+fc)/(rt+ft)*100:.0f}%'})")

    if rt > 0 and ft > 0:
        overall_acc = (rc + fc) / (rt + ft)
        if overall_acc >= 0.65:
            print("\n  [OK] Model is working correctly (>= 65% accuracy on test samples)")
        else:
            print("\n  [!] Accuracy below 65% - consider adding real-world images to dataset")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 64)
    print("  Deepfake Detection -- Improved Retrain Pipeline")
    print("=" * 64)

    step_dataset()
    trainer = step_train()
    step_evaluate(trainer)
    step_smoke_test()

    print("\n" + "=" * 64)
    print("  [OK] Retraining complete!")
    print("  Model saved to: models/deepfake_model.h5")
    print("  Restart the Django server to load the new model.")
    print("=" * 64 + "\n")
