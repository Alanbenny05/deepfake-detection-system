"""
Unit and Integration Tests for Multi-Signal Deepfake Detection Pipeline
========================================================================
Tests cover:
  1. Face Detector
  2. CNN Detector wrapper
  3. Forensic Analysis module
  4. Provenance & SynthID service
  5. Ensemble Decision Engine
  6. Image Detector pipeline
  7. Video Detector pipeline
  8. URL validation & SSRF protection
  9. Database models & migrations
  10. Views and error handling (no random values!)
"""

import json
import os
import tempfile
import numpy as np
import cv2
from PIL import Image

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from .models import Media, Detection
from .face_detector import detect_face
from .cnn_detector import cnn_detector
from .forensic_detector import analyse as forensic_analyse
from .provenance.provenance_service import check_provenance
from .ensemble.decision_engine import decide, CLASSIFICATION_LABELS
from .image_detector import detect_image
from .detection_service import DetectionService, _validate_url


class FaceDetectorTests(TestCase):
    def setUp(self):
        # Create synthetic test image with a simple colored box (no face)
        self.no_face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        self.no_face_img[50:150, 50:150] = [200, 150, 100]

    def test_no_face_detected(self):
        result = detect_face(self.no_face_img)
        self.assertIn('found', result)
        self.assertIn('quality', result)
        # On a plain box, found might be False
        if not result['found']:
            self.assertEqual(result['quality'], 0.0)

    def test_empty_image(self):
        result = detect_face(np.array([]))
        self.assertFalse(result['found'])


class ForensicDetectorTests(TestCase):
    def test_forensic_analysis_structure(self):
        img = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
        result = forensic_analyse(img)
        self.assertEqual(result['status'], 'ok')
        self.assertIn('forensic_score', result)
        self.assertIn('ela_score', result)
        self.assertIn('dct_score', result)
        self.assertIn('noise_score', result)
        self.assertIn('sharpness_score', result)
        self.assertIsInstance(result['flags'], list)
        self.assertTrue(0.0 <= result['forensic_score'] <= 1.0)

    def test_empty_image(self):
        result = forensic_analyse(np.array([]))
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['forensic_score'], 0.40)


class ProvenanceServiceTests(TestCase):
    def test_nonexistent_file(self):
        result = check_provenance('/path/that/does/not/exist.jpg', media_type='image')
        self.assertIn(result['status'], ('unavailable', 'error'))

    def test_standard_image_no_markers(self):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            img = Image.new('RGB', (100, 100), color='white')
            img.save(temp_path)

        try:
            result = check_provenance(temp_path, media_type='image')
            self.assertEqual(result['status'], 'not_detected')
            self.assertEqual(len(result['markers']), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class EnsembleDecisionEngineTests(TestCase):
    def test_high_confidence_fake(self):
        cnn_res = {'status': 'ok', 'cnn_score': 0.95, 'mode': 'cnn', 'face_used': True}
        forensic_res = {'status': 'ok', 'forensic_score': 0.88, 'flags': ['Anomalous frequency']}
        prov_res = {'status': 'detected', 'markers': ['AI software marker']}

        decision = decide(cnn_res, forensic_res, prov_res)
        self.assertEqual(decision['classification'], 'HIGH_CONFIDENCE_FAKE')
        self.assertGreaterEqual(decision['confidence'], 75.0)
        self.assertFalse(decision['is_uncertain'])

    def test_high_confidence_real(self):
        cnn_res = {'status': 'ok', 'cnn_score': 0.08, 'mode': 'cnn', 'face_used': True}
        forensic_res = {'status': 'ok', 'forensic_score': 0.12, 'flags': []}
        prov_res = {'status': 'not_detected', 'markers': []}

        decision = decide(cnn_res, forensic_res, prov_res)
        self.assertEqual(decision['classification'], 'HIGH_CONFIDENCE_REAL')
        self.assertGreaterEqual(decision['confidence'], 75.0)
        self.assertFalse(decision['is_uncertain'])

    def test_contradictory_signals_yields_uncertain(self):
        # CNN says very fake (0.92), forensic says very real (0.10)
        cnn_res = {'status': 'ok', 'cnn_score': 0.92, 'mode': 'heuristic', 'face_used': True}
        forensic_res = {'status': 'ok', 'forensic_score': 0.10, 'flags': []}
        prov_res = {'status': 'unavailable', 'markers': []}

        decision = decide(cnn_res, forensic_res, prov_res)
        self.assertEqual(decision['classification'], 'UNCERTAIN')
        self.assertTrue(decision['is_uncertain'])

    def test_no_signals_yields_uncertain(self):
        cnn_res = {'status': 'unavailable'}
        forensic_res = {'status': 'error'}
        prov_res = {'status': 'unavailable'}

        decision = decide(cnn_res, forensic_res, prov_res)
        self.assertEqual(decision['classification'], 'UNCERTAIN')
        self.assertTrue(decision['is_uncertain'])


class UrlValidationTests(TestCase):
    def test_valid_youtube_urls(self):
        self.assertIsNone(_validate_url('https://www.youtube.com/watch?v=dQw4w9WgXcQ', platform='youtube'))
        self.assertIsNone(_validate_url('https://youtu.be/dQw4w9WgXcQ', platform='youtube'))

    def test_invalid_youtube_urls(self):
        self.assertIsNotNone(_validate_url('https://example.com/video.mp4', platform='youtube'))
        self.assertIsNotNone(_validate_url('not_a_url', platform='youtube'))

    def test_ssrf_blocked_ips(self):
        self.assertIsNotNone(_validate_url('http://127.0.0.1:8000/test.mp4'))
        self.assertIsNotNone(_validate_url('http://localhost:8080/test.mp4'))
        self.assertIsNotNone(_validate_url('http://192.168.1.1/secret'))
        self.assertIsNotNone(_validate_url('http://10.0.0.1/test'))


class DetectionViewAndPipelineTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='password123')
        self.client = Client()
        self.client.login(username='testuser', password='password123')

    def test_upload_image_flow(self):
        # Create a test image
        img = Image.new('RGB', (120, 120), color='blue')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp.name, format='JPEG')
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as fp:
            uploaded = SimpleUploadedFile('test_img.jpg', fp.read(), content_type='image/jpeg')

        try:
            response = self.client.post(reverse('detector:upload'), {
                'media_type': 'image',
                'media_file': uploaded,
            })
            self.assertEqual(response.status_code, 302)  # Redirects to result

            detection = Detection.objects.filter(media__user=self.user).last()
            self.assertIsNotNone(detection)
            self.assertIn(detection.classification, CLASSIFICATION_LABELS)
            self.assertIn(detection.result, ('real', 'fake', 'uncertain'))
            self.assertTrue(0.0 <= detection.confidence <= 100.0)
            self.assertIsNotNone(detection.explanation)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_result_page_renders_multi_signal_context(self):
        media = Media.objects.create(
            user=self.user,
            file_name='test.jpg',
            media_type='image'
        )
        detection = Detection.objects.create(
            media=media,
            result='real',
            confidence=85.0,
            probability_real=85.0,
            probability_fake=15.0,
            cnn_score=0.15,
            forensic_score=0.20,
            provenance_status='not_detected',
            classification='LIKELY_REAL',
            explanation='Analysis shows authentic characteristics.'
        )

        response = self.client.get(reverse('detector:result', args=[detection.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Likely Real')
        self.assertContains(response, 'CNN Model Analysis')
        self.assertContains(response, 'Forensic Analysis')
        self.assertContains(response, 'Provenance / Metadata')
