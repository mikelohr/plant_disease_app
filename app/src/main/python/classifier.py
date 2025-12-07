from tflite_runtime.interpreter import Interpreter
RUNTIME = "tflite_runtime"

import numpy as np
from PIL import Image, ImageOps
import os
import time

# Resolve resources relative to this file so paths are stable inside the APK.
BASE_DIR = os.path.dirname(__file__)
DEFAULT_MODEL = os.path.join(BASE_DIR, "mobilenet_model.tflite")
DEFAULT_LABELS = os.path.join(BASE_DIR, "labels.txt")

class LeafClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL, label_path: str = DEFAULT_LABELS, do_warmup: bool = True):
        start_time = time.time()
        print(f"[INFO] Initializing LeafClassifier ({RUNTIME})...")
        print(f"[DEBUG] Model path: {model_path}")
        print(f"[DEBUG] Label path: {label_path}")

        # Load TFLite model and allocate tensors
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # IO tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"[INFO] Model loaded.")
        print(f"[DEBUG] Input details: {self.input_details}")
        print(f"[DEBUG] Output details: {self.output_details}")

        # Labels
        self.labels = self.load_labels(label_path)
        self.verify_model_compatibility()

        # Optional warm-up to reduce first-inference latency
        if do_warmup:
            try:
                self._warmup()
            except Exception as e:
                print(f"[WARN] Warm-up failed: {e}")

        elapsed = time.time() - start_time
        print(f"[INFO] LeafClassifier initialized in {elapsed:.2f} s.")

    # ---------- Setup helpers ----------

    def load_labels(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            print(f"[INFO] Loaded {len(labels)} labels from {path}.")
            return labels
        except Exception as e:
            print(f"[ERROR] Failed to load labels '{path}': {e}")
            return []

    def get_input_size(self):
        shape = self.input_details[0]["shape"]  # e.g., [1, 224, 224, 3]
        if len(shape) == 4:
            return int(shape[1]), int(shape[2])
        raise ValueError(f"[ERROR] Unexpected input shape: {shape}")

    def verify_model_compatibility(self):
        dtype = self.input_details[0]["dtype"]
        print(f"[INFO] Model input dtype: {dtype}")
        if dtype != np.float32:
            print(f"[WARN] Non-float32 input. Converting to float32 in preprocessing.")
        # Channel checks could be added if needed.

    def _warmup(self):
        w, h = self.get_input_size()
        print(f"[INFO] Performing warm-up with dummy input {w}x{h}...")
        dummy = np.zeros((1, h, w, 3), dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], dummy)
        self.interpreter.invoke()
        print(f"[INFO] Warm-up complete.")

    # ---------- Image helpers ----------

    def open_image_rgb(self, path: str):
        img = Image.open(path)
        try:
            img = ImageOps.exif_transpose(img)  # correct orientation
        except Exception as e:
            print(f"[WARN] EXIF transpose failed: {e}")
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def is_valid_file(self, path: str):
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
            Image.open(path).verify()  # integrity check
            return True
        except Exception:
            return False

    def preprocess(self, image_path: str):
        print(f"[INFO] Preprocessing image: {image_path}")
        w, h = self.get_input_size()
        try:
            img = self.open_image_rgb(image_path).resize((w, h))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if arr.ndim == 2:  # grayscale edge case
                arr = np.stack([arr] * 3, axis=-1)
            arr = np.expand_dims(arr, axis=0)  # NHWC batch
            print(f"[DEBUG] Preprocessed shape: {arr.shape}, dtype: {arr.dtype}")
            return arr
        except Exception as e:
            print(f"[ERROR] Failed to preprocess image: {e}")
            raise

    # ---------- Inference helpers ----------

    def softmax_stable(self, logits):
        m = np.max(logits)
        exps = np.exp(logits - m)
        denom = np.sum(exps)
        if denom == 0.0:
            print("[WARN] Softmax denominator is zero; returning uniform probabilities.")
            return np.ones_like(logits) / len(logits)
        return exps / denom

    def top_k(self, probs, k=3):
        idxs = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in idxs]

    def map_label(self, idx: int):
        if 0 <= idx < len(self.labels):
            return self.labels[idx]
        print(f"[WARN] Label index out of range: {idx}")
        return f"Unknown (index {idx})"

    def apply_threshold(self, label: str, confidence: float, min_conf: float = 0.30):
        if confidence < min_conf:
            return "Uncertain", confidence
        return label, confidence

    # ---------- Public API ----------

    def predict(self, image_path: str):
        start = time.time()
        print(f"[INFO] Starting prediction for: {image_path}")

        if not self.is_valid_file(image_path):
            print(f"[ERROR] Invalid image file: {image_path}")
            return "File not found or invalid", 0.0

        try:
            input_data = self.preprocess(image_path)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            print(f"[DEBUG] Input tensor set. Running inference...")
            self.interpreter.invoke()
            print(f"[INFO] Inference complete.")

            output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            print(f"[DEBUG] Raw model output: {output}")

            probs = self.softmax_stable(output)
            print(f"[DEBUG] Softmax probabilities: {probs}")

            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])

            label = self.map_label(top_idx)
            label, confidence = self.apply_threshold(label, confidence, min_conf=0.30)

            top3 = self.top_k(probs, k=3)
            pretty_top3 = [(self.map_label(i), c) for i, c in top3]
            print(f"[RESULT] Top-3: {pretty_top3}")
            print(f"[RESULT] Predicted label: {label}, Confidence: {confidence:.4f}")

            elapsed = time.time() - start
            print(f"[INFO] Prediction elapsed: {elapsed:.3f}s")

            return label, confidence

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return "Error", 0.0
