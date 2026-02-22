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
 * Wraps MediaPipe FaceLandmarker in VIDEO mode (synchronous detection).
 *
 * VIDEO mode is used instead of LIVE_STREAM because LIVE_STREAM applies an
 * internal OneEuroFilter to landmark coordinates before firing the callback,
 * adding ~1 camera frame (~33ms) of positional lag that cannot be corrected
 * by our predictor.  VIDEO mode returns raw detection results synchronously
 * with no inter-frame temporal smoothing, giving fresher, more accurate
 * landmarks for the current frame.
 *
 * Call [detectForVideo] once per camera frame from the analysis thread.
 */
class FaceLandmarkerHelper(
    private val context: Context,
    private val modelPath: String = "face_landmarker.task",
    val onResult: (FloatArray) -> Unit,
    val onError: (Exception) -> Unit,
) {
    // Single-threaded executor serialises all access to the FaceLandmarker
    // object (not thread-safe) and the setup/close lifecycle.
    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var landmarker: FaceLandmarker? = null
    private val setupStarted = AtomicBoolean(false)

    /** Latest landmark flat array [x0,y0,z0, x1,y1,z1 …], or null. */
    val latestLandmarks = AtomicReference<FloatArray?>(null)

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
                    .setRunningMode(RunningMode.VIDEO)
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
                        .setRunningMode(RunningMode.VIDEO)
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
     * Submit a bitmap for synchronous detection.  [timestampMs] must strictly
     * increase with each call (use the camera hardware timestamp).
     *
     * Runs on the internal executor to serialise access to the FaceLandmarker.
     * [detectForVideo] blocks until inference is complete, then immediately
     * calls [handleResult] — no async callback queue, no internal smoothing.
     *
     * The caller owns [bitmap] and is responsible for recycling it.
     */
    fun detectForVideo(bitmap: Bitmap, timestampMs: Long): FloatArray? {
        val lm = landmarker ?: return null
        // Run directly on the calling thread (CameraManager's analysisExecutor).
        // This keeps processFrame() blocked for the full inference duration so
        // CameraX's KEEP_ONLY_LATEST drops intermediate frames naturally.
        // The previous executor.execute{} wrapper returned immediately, causing
        // every camera frame to be queued onto FaceLandmarkerHelper.executor —
        // an unbounded queue that grew without bound and made lag dramatically
        // worse as the predictor fell further and further behind.
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = lm.detectForVideo(mpImage, timestampMs)
            handleResult(result, timestampMs)
        } catch (e: Exception) {
            Log.w(TAG, "detectForVideo error", e)
            null
        }
    }

    private fun handleResult(result: FaceLandmarkerResult, timestampMs: Long): FloatArray? {
        if (result.faceLandmarks().isEmpty()) {
            latestLandmarks.set(null)
            return null
        }

        val landmarks = result.faceLandmarks()[0]
        if (landmarks.size < NUM_LANDMARKS) return null

        val raw = FloatArray(NUM_LANDMARKS * 3)
        for (i in 0 until NUM_LANDMARKS) {
            val lm = landmarks[i]
            raw[i * 3]     = lm.x()
            raw[i * 3 + 1] = lm.y()
            raw[i * 3 + 2] = lm.z()
        }

        latestLandmarks.set(raw)
        onResult(raw)
        return raw
    }

    private fun buildBaseOptions(delegate: Delegate): BaseOptions {
        val builder = BaseOptions.builder().setDelegate(delegate)
        return if (modelPath.startsWith("/")) {
            builder.setModelAssetPath(modelPath).build()
        } else {
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
