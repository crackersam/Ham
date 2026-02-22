package com.ham.app.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.ham.app.face.FaceLandmarkerHelper
import com.ham.app.render.MakeupGLSurfaceView
import java.util.concurrent.Executors

private const val TAG = "CameraManager"

/**
 * Binds CameraX to the front camera with three use cases:
 *  - [Preview]: feeds the [MakeupGLSurfaceView] via its SurfaceTexture
 *  - [ImageAnalysis]: feeds [FaceLandmarkerHelper] for landmark detection
 *  - [ImageCapture]: available for still photo capture (JPEG)
 *
 * Call [bind] once the GL surface texture is ready.
 */
class CameraManager(private val context: Context) {

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var timestampMs = 0L

    /** Held so processFrame can push analysis dimensions to the renderer. */
    private var glSurfaceView: MakeupGLSurfaceView? = null

    fun bind(
        lifecycleOwner: LifecycleOwner,
        glSurfaceView: MakeupGLSurfaceView,
        landmarkerHelper: FaceLandmarkerHelper,
    ) {
        this.glSurfaceView = glSurfaceView
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider

            val preview = Preview.Builder()
                .setTargetResolution(Size(1280, 720))
                .build()

            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(1280, 720))
                .build()
            imageCapture = capture

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                processFrame(imageProxy, landmarkerHelper)
            }

            val selector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                provider.unbindAll()

                // Bind without a ViewPort so both the Preview OES texture and the
                // ImageAnalysis frame cover the same full native camera frame.
                // When a ViewPort is used, CameraX applies an additional software
                // crop to ImageAnalysis (via cropRect) that is not reflected in the
                // SurfaceTexture.getTransformMatrix() used by the GL renderer,
                // causing the landmark coordinate space to diverge from the
                // displayed camera image and making makeup appear larger and
                // offset. Without ViewPort both use cases share the hardware's
                // native sensor crop, keeping them in sync.
                provider.bindToLifecycle(lifecycleOwner, selector, preview, capture, analysis)

                // Connect preview to GL surface texture
                val surfaceProvider = glSurfaceView.buildSurfaceProvider()
                preview.setSurfaceProvider(surfaceProvider)

            } catch (e: Exception) {
                Log.e(TAG, "CameraX bind failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun processFrame(imageProxy: ImageProxy, helper: FaceLandmarkerHelper) {
        try {
            val rawBitmap = imageProxy.toBitmap()

            // CameraX 1.3.4+ toBitmap() already applies cropRect and returns a
            // bitmap whose dimensions equal (crop.width, crop.height). Only
            // re-crop manually if the dimensions differ, so we don't attempt an
            // out-of-bounds Bitmap.createBitmap on an already-cropped buffer.
            val crop = imageProxy.cropRect
            val croppedBitmap = if (rawBitmap.width != crop.width() ||
                                     rawBitmap.height != crop.height()) {
                Bitmap.createBitmap(rawBitmap, crop.left, crop.top,
                    crop.width(), crop.height()).also { rawBitmap.recycle() }
            } else {
                rawBitmap
            }

            // ImageAnalysis delivers frames in the sensor's native orientation.
            // rotationDegrees is the clockwise rotation needed to align the image
            // with the display.  We must apply it before MediaPipe so that
            // landmarks are returned in portrait (display) coordinate space.
            val rotation = imageProxy.imageInfo.rotationDegrees
            val rotatedBitmap: Bitmap = if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(
                    croppedBitmap, 0, 0, croppedBitmap.width, croppedBitmap.height, m, true
                ).also { croppedBitmap.recycle() }
            } else {
                croppedBitmap
            }

            // Tell the renderer the exact dimensions of the analysis frame so it
            // can apply an aspect-ratio correction when converting landmarks to NDC.
            glSurfaceView?.renderer?.let {
                it.analysisWidth  = rotatedBitmap.width.toFloat()
                it.analysisHeight = rotatedBitmap.height.toFloat()
            }

            val ts = System.currentTimeMillis()
            // MediaPipe LIVE_STREAM requires strictly increasing timestamps.
            if (ts > timestampMs) timestampMs = ts else timestampMs++
            // detectLiveStream takes bitmap ownership and recycles it.
            helper.detectLiveStream(rotatedBitmap, timestampMs)
        } finally {
            imageProxy.close()
        }
    }

    fun unbind() {
        glSurfaceView = null
        cameraProvider?.unbindAll()
        analysisExecutor.shutdown()
    }
}
