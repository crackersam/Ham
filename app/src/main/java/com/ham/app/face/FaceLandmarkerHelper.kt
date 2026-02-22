package com.ham.app.face

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

private const val TAG = "FaceLandmarkerHelper"
private const val NUM_LANDMARKS = 478

/**
 * Wraps MediaPipe FaceLandmarker.
 *
 * Runs inference on a dedicated background thread and publishes smoothed
 * landmark results via [onResult]. Call [detectLiveStream] once per camera
 * frame from any thread.
 *
 * [modelPath] may be:
 *  - A bare filename like "face_landmarker.task" → loaded from assets
 *  - An absolute path → loaded from the filesystem (downloaded model)
 */
class FaceLandmarkerHelper(
    private val context: Context,
    private val modelPath: String = "face_landmarker.task",
    val onResult: (FloatArray) -> Unit,
    val onError: (Exception) -> Unit,
) {
    private val executor = Executors.newSingleThreadExecutor()
    private var landmarker: FaceLandmarker? = null
    private val setupStarted = AtomicBoolean(false)

    private val kalman = LandmarkKalmanFilter(NUM_LANDMARKS)
    private val motion = MotionDetector()

    /** Latest smoothed landmark flat array [x0,y0,z0, x1,y1,z1 …], or null. */
    val latestLandmarks = AtomicReference<FloatArray?>(null)

    /** Extrapolates landmark positions at render frame rate between detections. */
    val predictor = LandmarkPredictor(NUM_LANDMARKS)

    /**
     * Initialise the FaceLandmarker.  Safe to call multiple times – subsequent
     * calls are no-ops if setup has already been requested.
     */
    fun setup() {
        if (!setupStarted.compareAndSet(false, true)) return
        executor.execute {
            try {
                val baseOptions = buildBaseOptions(Delegate.GPU)

                val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setNumFaces(1)
                    .setMinFaceDetectionConfidence(0.5f)
                    .setMinFacePresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setResultListener { result, _ -> handleResult(result) }
                    .setErrorListener { e -> onError(e) }
                    .build()

                landmarker = FaceLandmarker.createFromOptions(context, options)
            } catch (e: Exception) {
                // GPU delegate may not be supported on all devices – fall back to CPU
                try {
                    val baseOptions = buildBaseOptions(Delegate.CPU)

                    val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                        .setBaseOptions(baseOptions)
                        .setNumFaces(1)
                        .setMinFaceDetectionConfidence(0.5f)
                        .setMinFacePresenceConfidence(0.5f)
                        .setMinTrackingConfidence(0.5f)
                        .setRunningMode(RunningMode.LIVE_STREAM)
                        .setResultListener { result, _ -> handleResult(result) }
                        .setErrorListener { err -> onError(err) }
                        .build()

                    landmarker = FaceLandmarker.createFromOptions(context, options)
                } catch (e2: Exception) {
                    Log.e(TAG, "FaceLandmarker init failed", e2)
                    onError(e2)
                }
            }
        }
    }

    /**
     * Submit a bitmap for async detection. Must be called with a timestamp
     * that strictly increases each call (use [System.currentTimeMillis]).
     *
     * The helper takes ownership of [bitmap] and recycles it after MediaPipe
     * has read the pixel data, preventing a use-after-recycle race if the
     * caller recycles it immediately on the analysis thread.
     */
    fun detectLiveStream(bitmap: Bitmap, timestampMs: Long) {
        val lm = landmarker ?: run {
            bitmap.recycle()
            return
        }
        executor.execute {
            try {
                val mpImage = BitmapImageBuilder(bitmap).build()
                // detectAsync reads pixel data synchronously before returning,
                // so it is safe to recycle the bitmap afterwards.
                lm.detectAsync(mpImage, timestampMs)
            } catch (e: Exception) {
                Log.w(TAG, "detectAsync error", e)
            } finally {
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        }
    }

    private fun handleResult(result: FaceLandmarkerResult) {
        if (result.faceLandmarks().isEmpty()) {
            latestLandmarks.set(null)
            motion.reset()
            kalman.reset()
            predictor.reset()
            return
        }

        val landmarks = result.faceLandmarks()[0]
        if (landmarks.size < NUM_LANDMARKS) return

        val raw = FloatArray(NUM_LANDMARKS * 3)
        for (i in 0 until NUM_LANDMARKS) {
            val lm = landmarks[i]
            raw[i * 3]     = lm.x()
            raw[i * 3 + 1] = lm.y()
            raw[i * 3 + 2] = lm.z()
        }

        // Adaptive smoothing: less smoothing when face moves fast
        val motionAmount = motion.detect(raw)
        val smoothed = if (motionAmount > 0.5f) {
            // Fast motion – skip Kalman for this frame to stay responsive
            raw
        } else {
            kalman.filter(raw)
        }

        predictor.update(smoothed, System.currentTimeMillis())
        latestLandmarks.set(smoothed)
        onResult(smoothed)
    }

    private fun buildBaseOptions(delegate: Delegate): BaseOptions {
        val builder = BaseOptions.builder().setDelegate(delegate)
        return if (modelPath.startsWith("/")) {
            // Absolute filesystem path (downloaded model cached in filesDir)
            builder.setModelAssetPath(modelPath).build()
        } else {
            // Relative path → loaded from app assets
            builder.setModelAssetPath(modelPath).build()
        }
    }

    fun close() {
        setupStarted.set(false)
        executor.execute {
            landmarker?.close()
            landmarker = null
        }
    }
}
