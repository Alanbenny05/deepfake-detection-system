# 🤖 AI-Powered Deepfake Detection System

A full-stack, state-of-the-art Deepfake Detection web application built with **Django 5**, **TensorFlow/Keras**, and **OpenCV**. The platform enables users to analyze images, video files, YouTube videos, and Instagram Reels to verify authenticity and detect AI-generated media manipulation.

---

## 🌟 Key Features

- **Multi-Source Media Analysis**:
  - 🖼️ **Image Detection**: Upload JPG, PNG, and WEBP images for instant forensic analysis.
  - 🎥 **Video Detection**: Frame-by-frame deep learning analysis of MP4, AVI, MOV, and MKV video files.
  - 📺 **YouTube Video Analysis**: Paste YouTube URLs to fetch and analyze video content.
  - 📸 **Instagram Reel Analysis**: Analyze Instagram Reels via public link.
- **Deep Learning Architecture**:
  - Facial extraction and alignment with OpenCV Haar Cascades.
  - Transfer Learning with **EfficientNetB0** for feature extraction and classification.
  - Probability scores (Real % vs. AI Generated %) with confidence metrics.
- **Interactive Analytics Dashboard**:
  - Real-time statistics: Total uploads, Real count, AI Generated count, and Accuracy rate.
  - Dynamic Chart.js distribution pie charts and 7-day trend analysis.
  - Recent activity history log.
- **PDF Forensic Reports**:
  - Automated downloadable PDF reports with detailed metadata, confidence metrics, and analysis remarks powered by ReportLab.
- **Authentication & User Management**:
  - Secure registration, login, logout, profile customization, password change, and password reset flows.
- **Admin Control Panel**:
  - Django administration dashboard for managing users, uploaded media, detection records, and generated reports.

---

## 🏗️ Project Architecture

```text
deepfake-detection-system/
├── dashboard/               # Analytics dashboard app
│   ├── urls.py
│   └── views.py
├── deepfake_detection/      # Core Django project configuration
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── detector/                # Deepfake detection core app
│   ├── admin.py             # Model admin configurations
│   ├── forms.py             # Upload & URL forms
│   ├── ml_model.py          # ML Inference engine (DeepfakeDetector)
│   ├── models.py            # Media, Detection, Report models
│   ├── tests.py             # Detector unit & integration tests
│   ├── utils.py             # PDF generation, video frame extraction, URL fetchers
│   └── views.py             # Upload, URL, Result, and Report views
├── history/                 # Detection history app
│   ├── urls.py
│   └── views.py
├── reports/                 # PDF report management app
│   ├── urls.py
│   └── views.py
├── users/                   # Authentication & user profile app
│   ├── admin.py             # User & Profile admin
│   ├── forms.py             # Registration, update forms
│   ├── models.py            # Profile model
│   ├── tests.py             # User authentication tests
│   └── views.py             # Auth & profile views
├── dataset/                 # Dataset folder for training/testing
│   ├── train/               # Training split (real/ & fake/)
│   ├── validation/          # Validation split (real/ & fake/)
│   └── test/                # Test split (real/ & fake/)
├── media/                   # User uploads and generated assets
│   ├── profile_pics/
│   ├── reports/
│   └── uploads/
├── models/                  # Saved weights and training artifacts (.h5)
├── static/                  # Static assets (CSS, JS, Fonts)
├── templates/               # HTML templates (Bootstrap 5)
├── check_dataset.py         # Utility to inspect dataset structure
├── generate_sample_dataset.py # Synthetic dataset generator for development
├── train_model.py           # Model training script
├── test_trained_model.py    # Model evaluation script
├── fix_and_train.py         # Automated environment repair & training script
├── manage.py                # Django CLI
├── requirements.txt         # Python dependencies
├── setup.bat                # Windows setup script
└── .env.example             # Environment configuration template
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ / 3.11
- Git

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Alanbenny05/deepfake-detection-system.git
cd deepfake-detection-system

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:

```bash
# Windows:
copy .env.example .env
# Linux/macOS:
cp .env.example .env
```

### 4. Database Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create a Superuser (Admin)

```bash
python manage.py createsuperuser
```

### 6. Generate Sample Dataset / Populate Data

```bash
# Generate sample test fixtures for development & testing:
python generate_sample_dataset.py

# Verify dataset structure:
python check_dataset.py
```

### 7. Run the Application

```bash
python manage.py runserver
```

Open your browser and navigate to: `http://127.0.0.1:8000/`

---

## 🧪 Running Automated Tests

Run the full Django test suite with:

```bash
python manage.py test users detector
```

---

## 🧠 Model Training

To train a deepfake detector model using your own dataset (e.g. Celeb-DF, FaceForensics++, or 140k Real & Fake Faces):

1. Organize your images into `dataset/train/real`, `dataset/train/fake`, `dataset/validation/real`, `dataset/validation/fake`.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. Evaluate the trained weights:
   ```bash
   python test_trained_model.py
   ```

---

## 🛡️ License

This project is open source and available under the [MIT License](LICENSE).
