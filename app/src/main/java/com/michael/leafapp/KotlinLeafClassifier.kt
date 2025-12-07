package com.michael.leafapp

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.nio.ByteBuffer

// Simple data holder for classification results
data class ClassificationResult(val label: String, val confidence: Float)

class KotlinLeafClassifier(private val context: Context) {

    companion object {
        private const val TAG = "LeafClassifier"
        private const val MODEL_FILE = "mobilenet_model_compat.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val INPUT_SIZE = 224 // adjust to your model’s expected input size
    }

    // Lazily initialize the Interpreter with options
    private val interpreter: Interpreter by lazy {
        val modelBuffer: ByteBuffer = FileUtil.loadMappedFile(context, MODEL_FILE)
        val options = Interpreter.Options().apply { setNumThreads(4) }
        Interpreter(modelBuffer, options)
    }

    // Load labels once from assets
    private val labels: List<String> by lazy {
        FileUtil.loadLabels(context, LABELS_FILE)
    }

    // Classify an image file given its path
    fun classifyImageFile(path: String): ClassificationResult? {
        val bitmap = BitmapFactory.decodeFile(path) ?: return null

        // Preprocess: resize + normalize
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // divide pixel values by 255 to scale [0,1]
            .build()

        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
        val inputBuffer = tensorImage.buffer

        // Allocate output array [1 x NUM_CLASSES]
        val output = Array(1) { FloatArray(labels.size) }

        try {
            interpreter.run(inputBuffer, output)
        } catch (e: Exception) {
            Log.e(TAG, "Interpreter run failed", e)
            return null
        }

        val confidences = output[0]
        val maxIndex = confidences.indices.maxByOrNull { confidences[it] } ?: return null
        return ClassificationResult(labels[maxIndex], confidences[maxIndex])
    }

    // Cleanup
    fun close() {
        interpreter.close()
    }
}
