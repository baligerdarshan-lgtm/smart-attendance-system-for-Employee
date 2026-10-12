"""
Face Recognition & Detection utilities using OpenCV YuNet (detection) and SFace (recognition).
Provides face detection, feature extraction, cosine similarity matching, and local embeddings management.
"""

import os
import glob
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Support serverless read-only filesystems (Vercel / AWS Lambda)
if os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
    FACE_DATA_DIR = os.path.join('/tmp', 'face_data')
    repo_face_data = os.path.join(BASE_DIR, 'face_data')
    if os.path.exists(repo_face_data) and not os.path.exists(FACE_DATA_DIR):
        try:
            import shutil
            shutil.copytree(repo_face_data, FACE_DATA_DIR, dirs_exist_ok=True)
        except Exception:
            pass
else:
    FACE_DATA_DIR = os.path.join(BASE_DIR, 'face_data')

# Cosine similarity threshold for SFace
# OpenCV documentation recommends 0.363 for SFace cosine matching.
COSINE_THRESHOLD = 0.380

class FaceEngine:
    _instance = None

    def __init__(self):
        self.detector = None
        self.recognizer = None
        self.embeddings: Dict[str, np.ndarray] = {}  # {employee_id: embedding_array}
        self.loaded_model_paths = {}
        self.init_models()
        self.load_all_embeddings()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FaceEngine()
        return cls._instance

    def find_model_file(self, pattern: str) -> Optional[str]:
        """Finds a model file matching a wildcard pattern in MODELS_DIR."""
        matches = glob.glob(os.path.join(MODELS_DIR, pattern))
        if matches:
            return matches[0]
        return None

    def init_models(self):
        """Initializes YuNet Face Detector and SFace Face Recognizer."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(FACE_DATA_DIR, exist_ok=True)

        # 1. Locate YuNet detector model
        yunet_path = self.find_model_file("face_detection_yunet*.onnx")
        if not yunet_path:
            # Fallback to direct name
            yunet_path = os.path.join(MODELS_DIR, "face_detection_yunet_2026may.onnx")
            if not os.path.exists(yunet_path):
                yunet_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

        # 2. Locate SFace recognizer model
        sface_path = self.find_model_file("face_recognition_sface*.onnx")
        if not sface_path:
            sface_path = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

        if not os.path.exists(yunet_path):
            print(f"[FaceEngine Error] YuNet model not found at {yunet_path}")
            return
        if not os.path.exists(sface_path):
            print(f"[FaceEngine Error] SFace model not found at {sface_path}")
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=yunet_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=sface_path,
                config=""
            )
            self.loaded_model_paths = {
                "yunet": yunet_path,
                "sface": sface_path
            }
            print(f"[FaceEngine] Successfully initialized YuNet ({os.path.basename(yunet_path)}) and SFace ({os.path.basename(sface_path)})")
        except Exception as e:
            print(f"[FaceEngine Exception] Failed initializing models: {e}")

    def load_all_embeddings(self):
        """Loads all employee face feature embeddings from FACE_DATA_DIR."""
        self.embeddings.clear()
        if not os.path.exists(FACE_DATA_DIR):
            return

        npy_files = glob.glob(os.path.join(FACE_DATA_DIR, "*.npy"))
        for filepath in npy_files:
            filename = os.path.basename(filepath)
            employee_id = os.path.splitext(filename)[0]
            try:
                emb = np.load(filepath)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                self.embeddings[employee_id] = emb
            except Exception as e:
                print(f"[FaceEngine] Error loading embedding for {employee_id}: {e}")
        print(f"[FaceEngine] Loaded {len(self.embeddings)} face embedding(s).")

    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces in the given BGR image.
        Returns the detection array for the primary (largest/highest confidence) face,
        or None if no face is detected.
        """
        if self.detector is None or image is None:
            return None

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return None

        # Sort by confidence score (index 14) and bounding box area
        sorted_faces = sorted(faces, key=lambda f: (f[14], f[2] * f[3]), reverse=True)
        return sorted_faces[0]

    def extract_feature(self, image: np.ndarray, face_info: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Extracts a 128D feature embedding vector from an image using SFace.
        If face_info is None, runs detection first.
        """
        if self.recognizer is None or image is None:
            return None

        if face_info is None:
            face_info = self.detect_face(image)
            if face_info is None:
                return None

        try:
            aligned_face = self.recognizer.alignCrop(image, face_info)
            feature = self.recognizer.feature(aligned_face)
            return feature
        except Exception as e:
            print(f"[FaceEngine] Feature extraction error: {e}")
            return None

    def register_employee_face(self, employee_id: str, image: np.ndarray) -> Tuple[bool, str]:
        """
        Extracts face embedding from image and saves it to face_data/<employee_id>.npy.
        Returns (success: bool, message: str).
        """
        if image is None or image.size == 0:
            return False, "Invalid image data."

        face_info = self.detect_face(image)
        if face_info is None:
            return False, "No face detected in the provided photo. Please ensure good lighting and face clearly visible."

        feature = self.extract_feature(image, face_info)
        if feature is None:
            return False, "Could not extract face features from image."

        # Save embedding file
        save_path = os.path.join(FACE_DATA_DIR, f"{employee_id}.npy")
        np.save(save_path, feature)
        self.embeddings[employee_id] = feature

        return True, "Face feature extracted and saved successfully."

    def delete_employee_face(self, employee_id: str):
        """Removes employee face embedding from memory and disk."""
        if employee_id in self.embeddings:
            del self.embeddings[employee_id]
        filepath = os.path.join(FACE_DATA_DIR, f"{employee_id}.npy")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"[FaceEngine] Error deleting face file for {employee_id}: {e}")

    def match_face(self, query_feature: np.ndarray) -> Tuple[Optional[str], float, float, bool]:
        """
        Matches a query feature against all loaded employee embeddings.
        Returns (best_employee_id, raw_cosine_score, calibrated_confidence_percentage, is_match).
        """
        if self.recognizer is None or query_feature is None or len(self.embeddings) == 0:
            return None, 0.0, 0.0, False

        best_id = None
        best_score = -1.0

        for emp_id, emb in self.embeddings.items():
            try:
                score = self.recognizer.match(query_feature, emb, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = float(score)
                    best_id = emp_id
            except Exception as e:
                print(f"[FaceEngine] Match error against {emp_id}: {e}")

        # Calibration:
        # Cosine score ranges from approx -1.0 to 1.0, with ~0.38 being the decision threshold.
        # Calibrate to a 0-100% confidence scale for readable UI:
        # If score >= COSINE_THRESHOLD, confidence scales smoothly from 75% to 99%.
        # If score < COSINE_THRESHOLD, confidence scales from 0% to 74%.
        if best_score >= COSINE_THRESHOLD:
            # Map [COSINE_THRESHOLD, 0.85] -> [75%, 99%]
            normalized = min(max((best_score - COSINE_THRESHOLD) / (0.85 - COSINE_THRESHOLD), 0.0), 1.0)
            calibrated_percent = round(75.0 + (normalized * 24.0), 1)
            is_match = True
        else:
            normalized = max(best_score / max(COSINE_THRESHOLD, 0.01), 0.0)
            calibrated_percent = round(normalized * 70.0, 1)
            is_match = False

        return best_id, best_score, calibrated_percent, is_match

    def recognize_from_bgr_frame(self, frame: np.ndarray) -> Dict:
        """
        Full pipeline for a video frame:
        1. Detect face(s)
        2. Extract features
        3. Match against employee database
        4. Return detection boxes and matching info
        """
        if self.detector is None or self.recognizer is None:
            return {
                "success": False,
                "error": "Face models not loaded properly on server.",
                "faces_detected": 0
            }

        face_info = self.detect_face(frame)
        if face_info is None:
            return {
                "success": False,
                "error": "No face detected in camera frame.",
                "faces_detected": 0
            }

        # Bounding box [x, y, w, h]
        x, y, w, h = int(face_info[0]), int(face_info[1]), int(face_info[2]), int(face_info[3])
        det_score = float(face_info[14])

        feature = self.extract_feature(frame, face_info)
        if feature is None:
            return {
                "success": False,
                "error": "Could not extract face characteristics.",
                "faces_detected": 1,
                "box": [x, y, w, h]
            }

        best_id, cosine_score, confidence_percent, is_match = self.match_face(feature)

        return {
            "success": True,
            "faces_detected": 1,
            "box": [x, y, w, h],
            "detection_score": round(det_score, 2),
            "matched_employee_id": best_id if is_match else None,
            "cosine_score": round(cosine_score, 4),
            "confidence_percent": confidence_percent,
            "is_match": is_match
        }
