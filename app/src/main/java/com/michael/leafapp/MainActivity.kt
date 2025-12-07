package com.michael.leafapp

import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.michael.leafapp.ui.theme.LeafAppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

private const val TAG = "LeafApp"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContent {
            LeafAppTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    LeafAppUI(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

/**
 * Helper: copy URI content to a temp file (runs on Dispatchers.IO).
 */
suspend fun uriToFile(context: android.content.Context, uri: android.net.Uri): File =
    withContext(Dispatchers.IO) {
        val cacheName = "leaf_temp_${System.currentTimeMillis()}.jpg"
        val tempFile = File(context.cacheDir, cacheName)
        Log.d(TAG, "Creating temp file: ${tempFile.absolutePath}")

        context.contentResolver.openInputStream(uri).use { input ->
            FileOutputStream(tempFile).use { output ->
                if (input == null) {
                    throw IllegalStateException("Failed to open input stream for URI: $uri")
                }
                val copied = input.copyTo(output)
                Log.d(TAG, "Copied $copied bytes to temp file")
            }
        }
        tempFile
    }

@Composable
fun LeafAppUI(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val activity = (context as? ComponentActivity)

    var resultText by remember { mutableStateOf("Tap to classify a leaf image") }
    var isLoading by remember { mutableStateOf(false) }
    var selectedUri by remember { mutableStateOf<Uri?>(null) }
    var tempFilePath by remember { mutableStateOf<String?>(null) }

    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            Log.d(TAG, "Picked image URI: $uri")
            selectedUri = uri
            isLoading = true

            activity?.lifecycleScope?.launch {
                try {
                    // File I/O off the UI thread
                    val file = uriToFile(context, uri)
                    tempFilePath = file.absolutePath
                    Log.d(TAG, "Temp file ready at: ${file.absolutePath}")

                    // Classification off the UI thread
                    val result = withContext(Dispatchers.Default) {
                        KotlinLeafClassifier(context).classifyImageFile(file.absolutePath)
                    }

                    result?.let {
                        resultText = "Disease: ${it.label} (${String.format("%.2f", it.confidence)})"
                        Log.d(TAG, "Classification result: $resultText")
                    } ?: run {
                        resultText = "Could not classify this image. Try another."
                        Log.e(TAG, "Classification returned null")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error during classification pipeline", e)
                    resultText = "Could not classify this image. Try another."
                } finally {
                    // Cleanup temp file
                    try {
                        tempFilePath?.let { path ->
                            val f = File(path)
                            if (f.exists()) {
                                val deleted = f.delete()
                                Log.d(TAG, "Temp file cleanup (${f.name}): $deleted")
                            }
                        }
                    } catch (cleanupErr: Exception) {
                        Log.w(TAG, "Temp file cleanup failed: ${cleanupErr.message}")
                    }
                    isLoading = false
                }
            }
        } else {
            Log.d(TAG, "No image selected")
            resultText = "No image selected."
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(text = resultText)

        selectedUri?.let { Text(text = "Selected: $it") }
        tempFilePath?.let { Text(text = "Temp file: $it") }

        if (isLoading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        Button(onClick = {
            Log.d(TAG, "Classify button clicked")
            pickImageLauncher.launch("image/*")
        }) {
            Text("Select and Classify Leaf")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun LeafAppPreview() {
    LeafAppTheme {
        LeafAppUI()
    }
}
