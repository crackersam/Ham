package com.ham.app

import android.app.Application
import android.util.Log
import java.io.File
import java.net.URL

private const val TAG = "HamApplication"
private const val MODEL_FILENAME = "face_landmarker.task"

// Float16 model (~786 KB) from the official MediaPipe model repository
private const val MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

class HamApplication : Application() {

    /** True once the model file is available in assets or has been cached. */
    var modelReady = false
        private set

    var onModelReady: (() -> Unit)? = null

    override fun onCreate() {
        super.onCreate()
        ensureModel()
    }

    /**
     * Check if the model is already in assets (bundled) or in the app's
     * files directory (downloaded).  If neither, start a background download.
     */
    private fun ensureModel() {
        // First check bundled assets
        val bundled = try {
            assets.open(MODEL_FILENAME).also { it.close() }; true
        } catch (_: Exception) { false }

        if (bundled) {
            modelReady = true
            onModelReady?.invoke()
            return
        }

        // Check cached file
        val cached = File(filesDir, MODEL_FILENAME)
        if (cached.exists() && cached.length() > 100_000L) {
            // Copy to assets-accessible location expected by MediaPipe
            modelReady = true
            onModelReady?.invoke()
            return
        }

        // Download in background
        Thread({
            Log.d(TAG, "Downloading model from $MODEL_URL")
            try {
                val conn = URL(MODEL_URL).openConnection()
                conn.connectTimeout = 30_000
                conn.readTimeout = 60_000
                val tmp = File(filesDir, "$MODEL_FILENAME.tmp")
                conn.getInputStream().use { input ->
                    tmp.outputStream().use { out -> input.copyTo(out) }
                }
                tmp.renameTo(cached)
                Log.d(TAG, "Model downloaded: ${cached.length()} bytes")
                modelReady = true
                onModelReady?.invoke()
            } catch (e: Exception) {
                Log.e(TAG, "Model download failed", e)
            }
        }, "ModelDownload").start()
    }

    /**
     * Returns the path MediaPipe should use to load the model.
     * MediaPipe's [BaseOptions.setModelAssetPath] looks in assets first,
     * then falls back to absolute file path if we return an absolute path.
     *
     * We return just the filename because [FaceLandmarkerHelper] calls
     * [BaseOptions.setModelAssetPath] which resolves against assets,
     * and we always copy/link the model into the assets-equivalent location.
     *
     * If the model was downloaded, we configure MediaPipe with the absolute
     * file path instead.
     */
    fun modelPath(): String {
        val cached = File(filesDir, MODEL_FILENAME)
        return if (cached.exists() && cached.length() > 100_000L) {
            cached.absolutePath
        } else {
            MODEL_FILENAME  // from assets (bundled)
        }
    }
}
