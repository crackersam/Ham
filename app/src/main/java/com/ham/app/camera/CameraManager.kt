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
import java.nio.ByteBuffer
import java.util.concurrent.Executors

private const val TAG = "CameraManager"

/**
 * Binds CameraX to the front camera with:
 *  - [ImageAnalysis]: feeds [FaceLandmarkerHelper] and the GL renderer from the same frames
 *  - [ImageCapture]: (optional) available for still photo capture (JPEG)
 *
 * The on-screen camera background is rendered from ImageAnalysis RGBA frames
 * uploaded to OpenGL, so we do not bind a CameraX Preview surface.
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

            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(1280, 720))
                .build()
            imageCapture = capture

            val analysis = ImageAnalysis.Builder()
                // 640×360 (16:9) matches the preview aspect ratio so both
                // use cases receive the same sensor crop from the camera HAL.
                // A 4:3 target (e.g. 320×240) would give a different field of
                // view than the 16:9 preview, making landmark y-positions
                // systematically wrong relative to what appears on screen.
                // MediaPipe internally resizes to 192×192, so 640×360 is
                // still far smaller than the original 1280×720 analysis feed.
                .setTargetResolution(Size(640, 360))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                processFrame(imageProxy, landmarkerHelper)
            }

            val selector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                provider.unbindAll()

                provider.bindToLifecycle(lifecycleOwner, selector, capture, analysis)

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

            // Use the camera hardware capture timestamp (CLOCK_MONOTONIC ns → ms).
            // This is the same clock used by SurfaceTexture.timestamp so the
            // predictor's time base aligns perfectly with the displayed OES frame,
            // removing any REALTIME/MONOTONIC clock conversion error.
            val hwMs = imageProxy.imageInfo.timestamp / 1_000_000L
            // MediaPipe LIVE_STREAM requires strictly increasing timestamps.
            if (hwMs > timestampMs) timestampMs = hwMs else timestampMs++
            val landmarks = helper.detectForVideo(rotatedBitmap, timestampMs)

            // Build an atomic per-frame packet for the renderer:
            // - RGBA pixels for the camera background
            // - Landmarks computed from the exact same pixels
            val sv = glSurfaceView
            val renderer = sv?.renderer
            if (renderer != null) {
                val w = rotatedBitmap.width
                val h = rotatedBitmap.height
                val bytes = w * h * 4
                val rgba: ByteBuffer? = renderer.acquireRgbaBuffer(bytes)
                if (rgba != null) {
                    rgba.clear()
                    rotatedBitmap.copyPixelsToBuffer(rgba)
                    rgba.rewind()
                    renderer.submitFrame(rgba, w, h, landmarks)
                    sv.requestRender()
                }
            }

            rotatedBitmap.recycle()
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
