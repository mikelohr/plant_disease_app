import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Load and preprocess dataset
def load_data(data_directory):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train = datagen.flow_from_directory(
        data_directory,
        target_size=(224, 224),
        subset='training',
        class_mode='categorical'
    )

    val = datagen.flow_from_directory(
        data_directory,
        target_size=(224, 224),
        subset='validation',
        class_mode='categorical'
    )

    return train, val

# Build MobileNetV2-based model
def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Train, evaluate, and export model
def train_and_export(data_directory, export_dir):
    train, val = load_data(data_directory)
    model = build_model(num_classes=train.num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Training on {train.samples} images across {train.num_classes} classes")
    history = model.fit(train, validation_data=val, epochs=10)

    # Save and reload model to validate
    model_path = os.path.join(export_dir, "model_check.h5")
    model.save(model_path)
    reloaded = tf.keras.models.load_model(model_path)

    loss, acc = reloaded.evaluate(val)
    print(f"Validation Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    if acc < 0.5:
        print("Warning: Model accuracy is low. Skipping TFLite export.")
        return

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(reloaded)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(export_dir, "mobilenet_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_path}")

    # Save labels in correct order
    sorted_labels = sorted(train.class_indices.items(), key=lambda x: x[1])
    labels_path = os.path.join(export_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for label, _ in sorted_labels:
            f.write(f"{label}\n")
    print(f"Labels saved to: {labels_path}")

if __name__ == "__main__":
    train_and_export(
        data_directory=r"C:\Users\mlohr\DeepLearning\anaconda3\plant_disease_app\assets",
        export_dir=r"C:\Users\mlohr\DeepLearning\anaconda3\plant_disease_app\model"
    )
