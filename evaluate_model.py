"""
Model and Pipeline Evaluation Script
====================================
Evaluates the CNN detector and the multi-signal ensemble pipeline on an
evaluation dataset (images / test samples).

Metrics computed from actual data:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC (when probabilities available)
  - Confusion Matrix
  - Classification Report

Also supports evaluating separate subsets (e.g. real vs fake, face quality).
NEVER fabricates evaluation metrics.
"""

import argparse
import os
import sys
import time

import numpy as np


def evaluate_dataset(dataset_dir, max_samples_per_class=None):
    """
    Evaluate detector on a dataset folder structured as:
       dataset_dir/
          real/
             img1.jpg, ...
          fake/
             img2.jpg, ...
    """
    real_dir = os.path.join(dataset_dir, 'real')
    fake_dir = os.path.join(dataset_dir, 'fake')

    if not (os.path.exists(real_dir) and os.path.exists(fake_dir)):
        print(f"Error: Dataset directory must contain 'real' and 'fake' subdirectories.")
        print(f"Looked for:\n  - {real_dir}\n  - {fake_dir}")
        return None

    # Collect files
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(valid_exts)]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(valid_exts)]

    if max_samples_per_class:
        real_files = real_files[:max_samples_per_class]
        fake_files = fake_files[:max_samples_per_class]

    total_samples = len(real_files) + len(fake_files)
    if total_samples == 0:
        print("No valid image files found in dataset directories.")
        return None

    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images (Total: {total_samples}).")
    print("Running evaluation across multi-signal pipeline...\n")

    # Import detector components
    from detector.image_detector import detect_image

    y_true = []
    y_pred_binary = []
    y_scores = []
    classifications = []
    processing_times = []

    # Process Real (Label: 0)
    for i, path in enumerate(real_files, 1):
        t0 = time.monotonic()
        res = detect_image(path)
        elapsed = time.monotonic() - t0
        processing_times.append(elapsed)

        score = res.get('final_score', 0.5)
        # Binary prediction: fake if score >= 0.5
        pred = 1 if score >= 0.50 else 0

        y_true.append(0)
        y_pred_binary.append(pred)
        y_scores.append(score)
        classifications.append(res.get('classification', 'UNCERTAIN'))
        print(f"[Real {i}/{len(real_files)}] Score: {score:.3f} | Class: {res.get('classification')} ({elapsed:.2f}s)")

    # Process Fake (Label: 1)
    for i, path in enumerate(fake_files, 1):
        t0 = time.monotonic()
        res = detect_image(path)
        elapsed = time.monotonic() - t0
        processing_times.append(elapsed)

        score = res.get('final_score', 0.5)
        pred = 1 if score >= 0.50 else 0

        y_true.append(1)
        y_pred_binary.append(pred)
        y_scores.append(score)
        classifications.append(res.get('classification', 'UNCERTAIN'))
        print(f"[Fake {i}/{len(fake_files)}] Score: {score:.3f} | Class: {res.get('classification')} ({elapsed:.2f}s)")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred_binary)
    y_scores = np.array(y_scores)

    # Compute actual metrics using sklearn
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'], zero_division=0)

    # Tier breakdown
    tier_counts = {}
    for c in classifications:
        tier_counts[c] = tier_counts.get(c, 0) + 1

    print("\n" + "="*60)
    print("           DEEPFAKE DETECTION EVALUATION REPORT           ")
    print("="*60)
    print(f"Total Samples Evaluated : {total_samples}")
    print(f"Mean Processing Time    : {np.mean(processing_times):.3f}s / image")
    print("-" * 60)
    print(f"Accuracy                : {acc * 100:.2f}%")
    print(f"Precision (Fake)        : {prec * 100:.2f}%")
    print(f"Recall (Fake)           : {rec * 100:.2f}%")
    print(f"F1-Score                : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC                 : {roc_auc:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print("                Predicted Real   Predicted Fake")
    print(f"  Actual Real :      {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"  Actual Fake :      {cm[1][0]:<15} {cm[1][1]:<15}")
    print("-" * 60)
    print("Classification Tiers Distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier:<22} : {count} ({count/total_samples*100:.1f}%)")
    print("-" * 60)
    print("Detailed Classification Report:")
    print(report)
    print("="*60)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tier_counts': tier_counts,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection pipeline on real dataset.")
    parser.add_argument('--dataset', type=str, default='dataset', help="Path to dataset root containing real/ and fake/")
    parser.add_argument('--max-samples', type=int, default=None, help="Max samples per class to evaluate")
    args = parser.parse_args()

    # Configure Django environment
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detection.settings')
    import django
    django.setup()

    evaluate_dataset(args.dataset, max_samples_per_class=args.max_samples)
