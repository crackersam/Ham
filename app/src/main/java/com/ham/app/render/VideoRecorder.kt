package com.ham.app.render

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.media.MediaRecorder
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.util.Log
import android.view.PixelCopy
import android.view.SurfaceView
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "VideoRecorder"
private const val FRAME_RATE = 30

/**
 * Records video of the composited GL output (camera + makeup overlay) by:
 *  1. Configuring [MediaRecorder] with [MediaRecorder.VideoSource.SURFACE]
 *  2. Capturing [GLSurfaceView] pixels with [PixelCopy] at 30 fps
 *  3. Drawing each captured frame onto the MediaRecorder's input [Surface]
 *  4. Saving the resulting MP4 to the gallery via MediaStore
 */
class VideoRecorder(private val context: Context) {

    var isRecording = false
        private set
    var onRecordingFinished: ((File?) -> Unit)? = null

    private var recorder: MediaRecorder? = null
    private var outputFile: File? = null
    private var recorderSurface: android.view.Surface? = null

    private val recording = AtomicBoolean(false)
    private val scheduler = Executors.newSingleThreadScheduledExecutor()
    private var frameTask: ScheduledFuture<*>? = null

    private val copyThread = HandlerThread("PixelCopyThread").also { it.start() }
    private val copyHandler = Handler(copyThread.looper)

    // ── Public API ────────────────────────────────────────────────────────────

    fun start(surfaceView: SurfaceView, width: Int, height: Int) {
        if (isRecording) return
        isRecording = true
        recording.set(true)

        val dir = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES)
            ?: context.cacheDir
        val file = File(dir, "ham_${System.currentTimeMillis()}.mp4")
        outputFile = file

        try {
            val mr = createRecorder(width, height, file)
            recorder = mr
            recorderSurface = mr.surface

            val intervalMs = (1000L / FRAME_RATE)
            frameTask = scheduler.scheduleAtFixedRate({
                if (recording.get()) captureAndDraw(surfaceView)
            }, 0L, intervalMs, TimeUnit.MILLISECONDS)

        } catch (e: Exception) {
            Log.e(TAG, "start failed", e)
            release()
            onRecordingFinished?.invoke(null)
        }
    }

    fun stop() {
        if (!isRecording) return
        isRecording = false
        recording.set(false)
        frameTask?.cancel(false)

        Thread.sleep(150) // let last frame flush

        try {
            recorder?.stop()
        } catch (e: Exception) {
            Log.e(TAG, "recorder stop error", e)
        }
        release()
        val file = outputFile
        if (file != null && file.exists() && file.length() > 0) {
            saveToGallery(file)
        } else {
            onRecordingFinished?.invoke(null)
        }
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private fun createRecorder(width: Int, height: Int, file: File): MediaRecorder {
        val mr = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(context)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }
        mr.apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoSize(width, height)
            setVideoFrameRate(FRAME_RATE)
            setVideoEncodingBitRate(8_000_000)
            setAudioEncodingBitRate(128_000)
            setAudioSamplingRate(44_100)
            setOutputFile(file.absolutePath)
            prepare()
            start()
        }
        return mr
    }

    private fun captureAndDraw(surfaceView: SurfaceView) {
        val surface = recorderSurface ?: return
        val w = surfaceView.width; val h = surfaceView.height
        if (w <= 0 || h <= 0) return

        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            PixelCopy.request(surfaceView, bmp, { result ->
                if (result == PixelCopy.SUCCESS) {
                    try {
                        val canvas: Canvas = surface.lockCanvas(null)
                        canvas.drawBitmap(bmp, 0f, 0f, null)
                        surface.unlockCanvasAndPost(canvas)
                    } catch (e: Exception) {
                        Log.w(TAG, "draw to recorder surface error", e)
                    }
                }
                bmp.recycle()
            }, copyHandler)
        } else {
            bmp.recycle()
        }
    }

    private fun release() {
        recorderSurface?.release()
        recorderSurface = null
        recorder?.release()
        recorder = null
    }

    private fun saveToGallery(file: File) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    put(MediaStore.Video.Media.RELATIVE_PATH,
                        "${Environment.DIRECTORY_MOVIES}/Ham")
                    put(MediaStore.Video.Media.IS_PENDING, 1)
                }
                val uri = context.contentResolver
                    .insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    context.contentResolver.openOutputStream(uri)?.use { out ->
                        file.inputStream().use { it.copyTo(out) }
                    }
                    values.clear()
                    values.put(MediaStore.Video.Media.IS_PENDING, 0)
                    context.contentResolver.update(uri, values, null, null)
                    file.delete()
                }
            }
            onRecordingFinished?.invoke(outputFile)
        } catch (e: Exception) {
            Log.e(TAG, "saveToGallery error", e)
            onRecordingFinished?.invoke(outputFile)
        }
    }
}
