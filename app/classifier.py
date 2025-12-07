import tensorflow as tf
import numpy as np
from PIL import Image

class LeafClassifier:
    def __init__(self, model_path, label_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = self.load_labels (label_path)
        
    def load_labels(self, path):
        with open(path, 'r', encoding="utf-8") as labels:
            loaded_labels = [line.strip() for line in labels if line.strip()]
            print("Loaded labels:", loaded_labels)
            print("Label count:", len(loaded_labels))
            return loaded_labels
    
    def preprocess(sef, image_path):
        img= Image.open(image_path).resize((224, 224))
        img = np.array(img)/ 255.0
        img = np.expand_dims(img.astype(np.float32), axis=0)
        return img
    
    def predict(self, image_path):
        input_data = self.preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

    # Apply softmax to get probabilities
        output = tf.nn.softmax(output).numpy()

    # Get top prediction
        top_idx = int(np.argmax(output))
        confidence = float(output[top_idx])

    # Defensive label mapping
        if top_idx < len(self.labels):
            label = self.labels[top_idx]
        else:
            label = f"Unknown (index {top_idx})"

        print("Top index:", top_idx)
        print("Confidence:", confidence)
        print("Label count:", len(self.labels))
        print("Mapped label:", repr(label))

        return label, confidence

    