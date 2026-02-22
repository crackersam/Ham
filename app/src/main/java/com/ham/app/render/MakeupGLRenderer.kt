package com.ham.app.render

import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.Log
import androidx.compose.ui.graphics.Color
import com.ham.app.R
import com.ham.app.data.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sqrt
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

private const val TAG = "MakeupGLRenderer"
private const val FLOAT_SIZE = 4
private const val STRIDE = 5 * FLOAT_SIZE   // position(2) + edgeFactor(1) + regionUV(2)

/**
 * Full OpenGL ES 2.0 renderer.
 *
 * Frame pipeline:
 *  1. Camera OES texture updated from [SurfaceTexture]
 *  2. Draw full-screen quad with camera shader (beauty filter + mirror)
 *  3. For each makeup region draw geometry with makeup shader (alpha-blend)
 */
class MakeupGLRenderer(private val context: Context) : GLSurfaceView.Renderer {

    private data class FramePacket(
        val rgba: ByteBuffer,
        val width: Int,
        val height: Int,
        val landmarks: FloatArray?,
    )

    // ── Camera texture (GL_TEXTURE_2D fed from ImageAnalysis RGBA frames) ─────
    private var camTextureId = 0
    private var camTexWidth = 0
    private var camTexHeight = 0

    // ── Camera shader ─────────────────────────────────────────────────────────
    private var camProg = 0
    private var camAPosition = 0
    private var camATexCoord = 0
    private var camUMirror = 0
    private var camUTone = 0
    private var camUTexture = 0

    // ── Foundation shader ─────────────────────────────────────────────────────
    // Draws the face-oval mesh sampling the OES camera texture so smoothing
    // and colour correction are applied only within the face boundary.
    private var foundProg = 0
    private var foundAPosition = 0
    private var foundAEdgeFactor = 0
    private var foundARegionUV = 0
    private var foundUTexture = 0
    private var foundUTexelSize = 0
    private var foundUSmooth = 0
    private var foundUTone = 0
    private var foundUFoundAlpha = 0
    private var foundUFoundTint = -1
    private var foundUFoundCoverage = -1
    private var foundUMirror = 0
    private var foundUDebugFoundation = -1
    private var foundUDebugColor = -1

    // ── Makeup shader ─────────────────────────────────────────────────────────
    private var mkProg = 0
    private var mkAPosition = 0
    private var mkAEdgeFactor = 0
    private var mkARegionUV = 0
    private var mkUColor = 0
    private var mkUColor2 = 0
    private var mkUBlendMode = 0
    private var mkUGradientDir = 0
    private var mkUEffectKind = 0
    private var mkUNoiseTex = 0
    private var mkUNoiseScale = 0
    private var mkUNoiseAmount = 0

    private var noiseTextureId = 0

    // ── Quad VBO ──────────────────────────────────────────────────────────────
    private var quadVbo = 0

    // ── Viewport ──────────────────────────────────────────────────────────────
    private var viewWidth = 1
    private var viewHeight = 1

    // ── State ─────────────────────────────────────────────────────────────────
    /** Latest smoothed landmark array from MediaPipe – set from any thread. */
    val latestLandmarks = AtomicReference<FloatArray?>(null)

    /** Current makeup style – set from any thread. */
    var currentStyle: MakeupStyle = MAKEUP_STYLES[0]

    /** Skin smoothing intensity (0–1). */
    var smoothIntensity = 0.45f
    var toneIntensity = 0.35f

    /** Whether we're currently mirroring (true for live preview). */
    var isMirrored = true

    /**
     * Debug foundation modes:
     * 0 = off, 1 = mask overlay, 2 = delta view (amplified |foundation - camera|).
     */
    @Volatile var debugFoundationMode: Int = 0

    /**
     * Dimensions of the most-recent analysis bitmap passed to MediaPipe
     * (post-cropRect, post-rotation).  Updated on the analysis thread but only
     * read on the GL thread during [onDrawFrame], so a slight lag is harmless.
     * Defaults to 1×1 so the correction ratio is 1.0 until the first frame.
     */
    @Volatile var analysisWidth:  Float = 1f
    @Volatile var analysisHeight: Float = 1f

    // ── Analysis-frame camera background (precise-lock path) ──────────────────
    // CameraManager can push an RGBA frame buffer + landmarks as an atomic unit.
    // The GL thread consumes the latest packet and uploads the RGBA bytes to a
    // GL texture, ensuring makeup is drawn on the exact same frame used for
    // landmark inference.
    private val pendingFrame = AtomicReference<FramePacket?>(null)
    private val rgbaPool = ArrayDeque<ByteBuffer>(2)
    private var rgbaPoolAllocated = 0

    /**
     * Acquire a direct RGBA buffer (w*h*4 bytes). Returns null if no buffer is
     * currently available (caller should drop the frame to avoid blocking).
     */
    @Synchronized
    fun acquireRgbaBuffer(requiredBytes: Int): ByteBuffer? {
        val it = rgbaPool.iterator()
        while (it.hasNext()) {
            val buf = it.next()
            if (buf.capacity() >= requiredBytes) {
                it.remove()
                buf.clear()
                return buf
            }
        }
        if (rgbaPoolAllocated >= 2) return null
        rgbaPoolAllocated++
        return ByteBuffer.allocateDirect(requiredBytes).apply { order(ByteOrder.nativeOrder()); clear() }
    }

    @Synchronized
    private fun releaseRgbaBuffer(buf: ByteBuffer) {
        if (rgbaPool.size < 2) rgbaPool.addLast(buf)
    }

    /**
     * Submit a new frame packet. If an older unconsumed packet exists, it is
     * dropped and its buffer returned to the pool.
     */
    fun submitFrame(rgba: ByteBuffer, width: Int, height: Int, landmarks: FloatArray?) {
        val old = pendingFrame.getAndSet(FramePacket(rgba, width, height, landmarks))
        if (old != null) releaseRgbaBuffer(old.rgba)
    }

    // ── Photo capture callback ────────────────────────────────────────────────
    var onPixelsReady: ((ByteArray, Int, Int) -> Unit)? = null
    private var captureNextFrame = false

    // ── Dynamic VBO for makeup ────────────────────────────────────────────────
    private var mkVbo = 0

    // ── Shared geometry staging buffer ────────────────────────────────────────
    // Pre-allocated once so drawGeometry() and drawFoundation() never call
    // ByteBuffer.allocateDirect() on the GL thread, eliminating GC pressure
    // that would otherwise cause micro-stutter (~20 allocations per frame).
    // 64 KB covers the largest mesh with headroom to spare.
    private val geomBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(65_536).apply { order(ByteOrder.nativeOrder()) }
    private val geomFloatBuffer: FloatBuffer = geomBuffer.asFloatBuffer()

    // ─────────────────────────────────────────────────────────────────────────
    // GLSurfaceView.Renderer callbacks
    // ─────────────────────────────────────────────────────────────────────────

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) = runGLSafely("onSurfaceCreated") {
        GLES20.glClearColor(0f, 0f, 0f, 1f)

        // Create 2D texture for camera frames (RGBA uploaded from ImageAnalysis)
        val texIds = IntArray(1)
        GLES20.glGenTextures(1, texIds, 0)
        camTextureId = texIds[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        // Compile shaders
        val camVert = loadRawString(context, R.raw.camera_vertex)
        val camFrag = loadRawString(context, R.raw.camera_fragment)
        camProg = linkProgram(camVert, camFrag)
        camAPosition     = GLES20.glGetAttribLocation(camProg, "aPosition")
        camATexCoord     = GLES20.glGetAttribLocation(camProg, "aTexCoord")
        camUMirror       = GLES20.glGetUniformLocation(camProg, "uMirror")
        camUTone         = GLES20.glGetUniformLocation(camProg, "uTone")
        camUTexture      = GLES20.glGetUniformLocation(camProg, "uTexture")

        val foundVert = loadRawString(context, R.raw.foundation_vertex)
        val foundFrag = loadRawString(context, R.raw.foundation_fragment)
        foundProg = linkProgram(foundVert, foundFrag)
        foundAPosition    = GLES20.glGetAttribLocation(foundProg,  "aPosition")
        foundAEdgeFactor  = GLES20.glGetAttribLocation(foundProg,  "aEdgeFactor")
        foundARegionUV    = GLES20.glGetAttribLocation(foundProg,  "aRegionUV")
        foundUTexture     = GLES20.glGetUniformLocation(foundProg,  "uTexture")
        foundUTexelSize   = GLES20.glGetUniformLocation(foundProg,  "uTexelSize")
        foundUSmooth      = GLES20.glGetUniformLocation(foundProg,  "uSmooth")
        foundUTone        = GLES20.glGetUniformLocation(foundProg,  "uTone")
        foundUFoundAlpha  = GLES20.glGetUniformLocation(foundProg,  "uFoundAlpha")
        foundUFoundTint   = GLES20.glGetUniformLocation(foundProg,  "uFoundTint")
        foundUFoundCoverage = GLES20.glGetUniformLocation(foundProg, "uFoundCoverage")
        foundUMirror      = GLES20.glGetUniformLocation(foundProg,  "uMirror")
        foundUDebugFoundation = GLES20.glGetUniformLocation(foundProg, "uDebugFoundation")
        foundUDebugColor = GLES20.glGetUniformLocation(foundProg, "uDebugColor")

        val mkVert = loadRawString(context, R.raw.makeup_vertex)
        val mkFrag = loadRawString(context, R.raw.makeup_fragment)
        mkProg = linkProgram(mkVert, mkFrag)
        mkAPosition  = GLES20.glGetAttribLocation(mkProg, "aPosition")
        mkAEdgeFactor = GLES20.glGetAttribLocation(mkProg, "aEdgeFactor")
        mkARegionUV  = GLES20.glGetAttribLocation(mkProg, "aRegionUV")
        mkUColor     = GLES20.glGetUniformLocation(mkProg, "uColor")
        mkUColor2    = GLES20.glGetUniformLocation(mkProg, "uColor2")
        mkUBlendMode = GLES20.glGetUniformLocation(mkProg, "uBlendMode")
        mkUGradientDir = GLES20.glGetUniformLocation(mkProg, "uGradientDir")
        mkUEffectKind = GLES20.glGetUniformLocation(mkProg, "uEffectKind")
        mkUNoiseTex = GLES20.glGetUniformLocation(mkProg, "uNoiseTex")
        mkUNoiseScale = GLES20.glGetUniformLocation(mkProg, "uNoiseScale")
        mkUNoiseAmount = GLES20.glGetUniformLocation(mkProg, "uNoiseAmount")

        // Full-screen quad: position(xy) + texCoord(uv).
        //
        // Android bitmaps store rows top→bottom, but OpenGL texture coords
        // conventionally treat v=0 as the bottom row.  Using v=1 at the screen
        // bottom (and v=0 at the top) flips the image vertically so the camera
        // background appears upright without having to vertically flip pixels.
        val quadData = floatArrayOf(
            // x     y     u     v
            -1f, -1f,  0f,  1f,   // screen bottom-left  → tex top-left
             1f, -1f,  1f,  1f,   // screen bottom-right → tex top-right
            -1f,  1f,  0f,  0f,   // screen top-left     → tex bottom-left
             1f,  1f,  1f,  0f,   // screen top-right    → tex bottom-right
        )
        val vbos = IntArray(2)
        GLES20.glGenBuffers(2, vbos, 0)
        quadVbo = vbos[0]
        mkVbo   = vbos[1]
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        val quadBuf = ByteBuffer.allocateDirect(quadData.size * FLOAT_SIZE)
            .apply { order(ByteOrder.nativeOrder()) }
            .asFloatBuffer()
            .apply { put(quadData); position(0) }
        GLES20.glBufferData(
            GLES20.GL_ARRAY_BUFFER,
            quadData.size * FLOAT_SIZE,
            quadBuf,
            GLES20.GL_STATIC_DRAW,
        )

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        // Small repeatable noise texture for blush grain (generated once).
        noiseTextureId = createNoiseTexture(size = 64, seed = 1337)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) = runGLSafely("onSurfaceChanged") {
        viewWidth = width
        viewHeight = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) = runGLSafely("onDrawFrame") {
        // Consume the latest analysis frame (if any) and upload it to the 2D texture.
        val packet = pendingFrame.getAndSet(null)
        if (packet != null) {
            uploadCameraTexture(packet)
            latestLandmarks.set(packet.landmarks)
            releaseRgbaBuffer(packet.rgba)
        }

        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        // ── Pass 1: Camera ────────────────────────────────────────────────────
        if (camTexWidth > 0 && camTexHeight > 0) {
            GLES20.glUseProgram(camProg)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)

            GLES20.glUniform1i(camUTexture, 0)
            GLES20.glUniform1f(camUMirror, if (isMirrored) 1f else 0f)
            GLES20.glUniform1f(camUTone, toneIntensity)

            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
            val stride4 = 4 * FLOAT_SIZE
            GLES20.glEnableVertexAttribArray(camAPosition)
            GLES20.glVertexAttribPointer(camAPosition, 2, GLES20.GL_FLOAT, false, stride4, 0)
            GLES20.glEnableVertexAttribArray(camATexCoord)
            GLES20.glVertexAttribPointer(camATexCoord, 2, GLES20.GL_FLOAT, false, stride4, 2 * FLOAT_SIZE)

            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        }

        // ── Pass 2: Makeup ────────────────────────────────────────────────────
        val style = currentStyle
        val lm = packet?.landmarks ?: latestLandmarks.get()
        val hasAnyLayer =
            style.foundationAlpha > 0.01f ||
                style.blushAlpha > 0.01f ||
                style.eyeshadowAlpha > 0.01f ||
                style.linerAlpha > 0.01f ||
                style.lipAlpha > 0.01f ||
                style.highlightAlpha > 0.01f ||
                style.browAlpha > 0.01f ||
                style.browFillAlpha > 0.01f ||
                style.lashAlpha > 0.01f
        if (lm != null && hasAnyLayer) {
            drawMakeup(lm, style)
        }

        // ── Photo capture ─────────────────────────────────────────────────────
        if (captureNextFrame) {
            captureNextFrame = false
            readPixels()
        }
    }

    private fun uploadCameraTexture(packet: FramePacket) {
        // Upload RGBA to GL_TEXTURE_2D. We reuse the texture allocation when the
        // frame size stays constant (normal case).
        val w = packet.width
        val h = packet.height
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)

        if (w != camTexWidth || h != camTexHeight) {
            camTexWidth = w
            camTexHeight = h
            GLES20.glTexImage2D(
                GLES20.GL_TEXTURE_2D,
                0,
                GLES20.GL_RGBA,
                w,
                h,
                0,
                GLES20.GL_RGBA,
                GLES20.GL_UNSIGNED_BYTE,
                null,
            )
        }

        packet.rgba.position(0)
        GLES20.glTexSubImage2D(
            GLES20.GL_TEXTURE_2D,
            0,
            0,
            0,
            w,
            h,
            GLES20.GL_RGBA,
            GLES20.GL_UNSIGNED_BYTE,
            packet.rgba,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Makeup drawing
    // ─────────────────────────────────────────────────────────────────────────

    private fun drawMakeup(lm: FloatArray, style: MakeupStyle) {
        GLES20.glEnable(GLES20.GL_BLEND)

        val mirror = isMirrored

        // After removing the ViewPort both the camera Preview OES texture and the
        // ImageAnalysis bitmap cover the same full native sensor frame.  The camera
        // stretches that frame to fill the viewport, and the landmark [0,1]→[-1,1]
        // NDC mapping stretches in exactly the same proportion, so no x-axis
        // correction is needed.  aspectScale is kept at 1f (no-op) to preserve the
        // parameter plumbing in MakeupGeometry for potential future use.
        val aspectScale = 1f

        // ── Foundation ────────────────────────────────────────────────────────
        // Drawn first so all subsequent makeup layers composite on top.
        // Samples the camera texture within the face-oval mesh boundary,
        // applying skin-smoothing and warmth correction only where the face is.
        if (style.foundationAlpha > 0.01f || debugFoundationMode != 0) {
            GLES20.glUseProgram(foundProg)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

            GLES20.glUniform1i(foundUTexture, 0)
            val texW = if (camTexWidth > 0) camTexWidth else viewWidth
            val texH = if (camTexHeight > 0) camTexHeight else viewHeight
            GLES20.glUniform2f(foundUTexelSize, 1f / texW.toFloat(), 1f / texH.toFloat())
            GLES20.glUniform1f(foundUSmooth, smoothIntensity)
            GLES20.glUniform1f(foundUTone, toneIntensity)
            val alpha = if (debugFoundationMode != 0) maxOf(style.foundationAlpha, 0.65f) else style.foundationAlpha
            GLES20.glUniform1f(foundUFoundAlpha, alpha)
            GLES20.glUniform1f(foundUMirror, if (mirror) 1f else 0f)

            if (foundUFoundTint >= 0) {
                GLES20.glUniform3f(foundUFoundTint, style.foundationTint.red, style.foundationTint.green, style.foundationTint.blue)
            }
            if (foundUFoundCoverage >= 0) {
                GLES20.glUniform1f(foundUFoundCoverage, style.foundationCoverage.coerceIn(0f, 1f))
            }

            if (foundUDebugFoundation >= 0) {
                GLES20.glUniform1f(foundUDebugFoundation, debugFoundationMode.toFloat())
            }
            if (foundUDebugColor >= 0) {
                GLES20.glUniform3f(foundUDebugColor, 0.15f, 1.0f, 0.25f)
            }

            drawFoundation(MakeupGeometry.buildFanMesh(lm, FACE_OVAL, mirror, aspectScale))
        }

        GLES20.glUseProgram(mkProg)

        // Bind grain noise texture (used only for blush; uniforms default to disabled).
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, noiseTextureId)
        GLES20.glUniform1i(mkUNoiseTex, 1)
        GLES20.glUniform1f(mkUNoiseScale, 12.0f)
        GLES20.glUniform1f(mkUNoiseAmount, 0f)
        GLES20.glUniform1f(mkUEffectKind, 0f)

        // ── Brows ─────────────────────────────────────────────────────────────
        if (style.browAlpha > 0.01f || style.browFillAlpha > 0.01f) {
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glUniform1f(mkUBlendMode, 1f)  // multiply-like for brow shading
            GLES20.glUniform1f(mkUGradientDir, 0f)

            val eyeH =
                (MakeupGeometry.eyeHeight(lm, LEFT_UPPER_LID_ARC,  LandmarkIndex.LEFT_EYE_LOWER,  mirror, aspectScale) +
                 MakeupGeometry.eyeHeight(lm, RIGHT_UPPER_LID_ARC, LandmarkIndex.RIGHT_EYE_LOWER, mirror, aspectScale)) * 0.5f

            // Pass A: soft powder-like fill
            if (style.browFillAlpha > 0.01f) {
                setMkColor(style.browColor, style.browFillAlpha)
                // Use a ribbon mesh between the two eyebrow contours to prevent
                // triangle fan spill below the underside edge.
                drawGeometry(MakeupGeometry.buildBrowRibbonMesh(lm, LEFT_BROW, mirror, aspectScale))
                drawGeometry(MakeupGeometry.buildBrowRibbonMesh(lm, RIGHT_BROW, mirror, aspectScale))
            }

            // Pass B: defined shaping
            if (style.browAlpha > 0.01f) {
                val defW = (eyeH * 0.06f).coerceIn(0.0025f, 0.012f)
                setMkColor(style.browColor, style.browAlpha)
                drawGeometry(
                    MakeupGeometry.buildStrokeMesh2D(
                        // Track the underside contour for a cleaner lower edge.
                        MakeupGeometry.buildBrowUndersidePath2D(lm, LEFT_BROW, mirror, aspectScale),
                        defW
                    )
                )
                drawGeometry(
                    MakeupGeometry.buildStrokeMesh2D(
                        MakeupGeometry.buildBrowUndersidePath2D(lm, RIGHT_BROW, mirror, aspectScale),
                        defW
                    )
                )
            }
        }

        // ── Blush ─────────────────────────────────────────────────────────────
        if (style.blushAlpha > 0.01f) {
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            val lCx0 = MakeupGeometry.lmX(lm, LandmarkIndex.LEFT_CHEEK_CENTER, mirror, aspectScale)
            val lCy0 = MakeupGeometry.lmY(lm, LandmarkIndex.LEFT_CHEEK_CENTER)
            val rCx0 = MakeupGeometry.lmX(lm, LandmarkIndex.RIGHT_CHEEK_CENTER, mirror, aspectScale)
            val rCy0 = MakeupGeometry.lmY(lm, LandmarkIndex.RIGHT_CHEEK_CENTER)

            // Radii derived from the actual bounding extent of each cheek's landmarks.
            val (lRx, lRy) = MakeupGeometry.landmarkBoundingRadius(lm, LEFT_CHEEK, mirror, aspectScale)
            val (rRx, rRy) = MakeupGeometry.landmarkBoundingRadius(lm, RIGHT_CHEEK, mirror, aspectScale)

            // Bias slightly up/out along the cheekbone direction (lip corner -> outer eye).
            val lEx = MakeupGeometry.lmX(lm, LandmarkIndex.LEFT_EYE_OUTER, mirror, aspectScale)
            val lEy = MakeupGeometry.lmY(lm, LandmarkIndex.LEFT_EYE_OUTER)
            val lMx = MakeupGeometry.lmX(lm, LandmarkIndex.LIP_LEFT, mirror, aspectScale)
            val lMy = MakeupGeometry.lmY(lm, LandmarkIndex.LIP_LEFT)
            val rEx = MakeupGeometry.lmX(lm, LandmarkIndex.RIGHT_EYE_OUTER, mirror, aspectScale)
            val rEy = MakeupGeometry.lmY(lm, LandmarkIndex.RIGHT_EYE_OUTER)
            val rMx = MakeupGeometry.lmX(lm, LandmarkIndex.LIP_RIGHT, mirror, aspectScale)
            val rMy = MakeupGeometry.lmY(lm, LandmarkIndex.LIP_RIGHT)

            fun biasedCenter(
                cx: Float,
                cy: Float,
                ex: Float,
                ey: Float,
                mx: Float,
                my: Float,
                rx: Float,
                ry: Float,
            ): Pair<Float, Float> {
                var dx = ex - mx
                var dy = ey - my
                val d = sqrt(dx * dx + dy * dy).coerceAtLeast(1e-6f)
                dx /= d; dy /= d
                val shift = 0.18f * min(rx, ry)
                return Pair(cx + dx * shift, cy + dy * shift)
            }

            val (lCx, lCy) = biasedCenter(lCx0, lCy0, lEx, lEy, lMx, lMy, lRx, lRy)
            val (rCx, rCy) = biasedCenter(rCx0, rCy0, rEx, rEy, rMx, rMy, rRx, rRy)

            GLES20.glUniform1f(mkUBlendMode, 0f)
            GLES20.glUniform1f(mkUGradientDir, 0f)
            GLES20.glUniform1f(mkUEffectKind, 1f)
            GLES20.glUniform1f(mkUNoiseAmount, 0.12f)

            setMkColor(style.blushColor, style.blushAlpha * 0.55f)
            drawGeometry(MakeupGeometry.buildBlushMesh(lCx, lCy, lRx, lRy, segments = 40))
            drawGeometry(MakeupGeometry.buildBlushMesh(rCx, rCy, rRx, rRy, segments = 40))

            // Second softer pass for depth
            setMkColor(style.blushColor, style.blushAlpha * 0.24f)
            drawGeometry(MakeupGeometry.buildBlushMesh(lCx, lCy, lRx * 1.35f, lRy * 1.30f, segments = 40))
            drawGeometry(MakeupGeometry.buildBlushMesh(rCx, rCy, rRx * 1.35f, rRy * 1.30f, segments = 40))

            // Reset defaults for subsequent layers.
            GLES20.glUniform1f(mkUEffectKind, 0f)
            GLES20.glUniform1f(mkUNoiseAmount, 0f)
        }

        // ── Eyeshadow ─────────────────────────────────────────────────────────
        if (style.eyeshadowAlpha > 0.01f) {
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glUniform1f(mkUGradientDir, 1f)
            GLES20.glUniform1f(mkUBlendMode, 0f)

            // Darker outer stop so colour deepens toward the outer corner
            val darker = darkenColor(style.eyeshadowColor, 0.55f)

            // expansionFactor: how far above the lash-line arc to push the
            // outer shadow boundary, expressed as a fraction of the arc's apex
            // radius.  Derived from the measured eye height so the shadow
            // scales with the actual eye size rather than a fixed constant.
            val leftExpansion = (MakeupGeometry.eyeHeight(lm, LEFT_UPPER_LID_ARC,  LandmarkIndex.LEFT_EYE_LOWER,  mirror, aspectScale) /
                                 MakeupGeometry.arcApexRadius(lm, LEFT_UPPER_LID_ARC,  mirror, aspectScale)).coerceIn(0.3f, 0.9f)
            val rightExpansion = (MakeupGeometry.eyeHeight(lm, RIGHT_UPPER_LID_ARC, LandmarkIndex.RIGHT_EYE_LOWER, mirror, aspectScale) /
                                  MakeupGeometry.arcApexRadius(lm, RIGHT_UPPER_LID_ARC, mirror, aspectScale)).coerceIn(0.3f, 0.9f)

            // Upper-eyelid-only strip mesh: sits above the lash line, never
            // covers the eyeball.
            val leftMesh  = MakeupGeometry.buildUpperEyelidMesh(lm, LEFT_UPPER_LID_ARC,  mirror, leftExpansion, aspectScale)
            val rightMesh = MakeupGeometry.buildUpperEyelidMesh(lm, RIGHT_UPPER_LID_ARC, mirror, rightExpansion, aspectScale)

            // Primary pigment pass
            setMkColor4(style.eyeshadowColor, style.eyeshadowAlpha * 0.90f,
                        darker, style.eyeshadowAlpha)
            drawGeometry(leftMesh)
            drawGeometry(rightMesh)

            // Soft secondary pass for blended depth
            setMkColor4(style.eyeshadowColor, style.eyeshadowAlpha * 0.38f,
                        darker, style.eyeshadowAlpha * 0.50f)
            drawGeometry(leftMesh)
            drawGeometry(rightMesh)
        }

        // ── Eyeliner ──────────────────────────────────────────────────────────
        if (style.linerAlpha > 0.01f) {
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glUniform1f(mkUBlendMode, 1f)  // multiply-like
            GLES20.glUniform1f(mkUGradientDir, 0f)

            // Width scaled to the actual measured height of each eye so liner
            // stays proportional regardless of face size or eye shape.
            val leftLinerW  = MakeupGeometry.eyeHeight(lm, LEFT_UPPER_LID_ARC,  LandmarkIndex.LEFT_EYE_LOWER,  mirror, aspectScale) * 0.10f
            val rightLinerW = MakeupGeometry.eyeHeight(lm, RIGHT_UPPER_LID_ARC, LandmarkIndex.RIGHT_EYE_LOWER, mirror, aspectScale) * 0.10f

            // Pass 1: soft shadow base
            setMkColor(style.linerColor, style.linerAlpha * 0.14f)
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, LEFT_LINER,  leftLinerW  * 2.0f, mirror, aspectScale))
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, RIGHT_LINER, rightLinerW * 2.0f, mirror, aspectScale))

            // Pass 2: mid pigment
            setMkColor(style.linerColor, style.linerAlpha * 0.28f)
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, LEFT_LINER,  leftLinerW  * 1.2f, mirror, aspectScale))
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, RIGHT_LINER, rightLinerW * 1.2f, mirror, aspectScale))

            // Pass 3: defined sharp line
            setMkColor(style.linerColor, style.linerAlpha * 0.30f)
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, LEFT_LINER,  leftLinerW  * 0.65f, mirror, aspectScale))
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, RIGHT_LINER, rightLinerW * 0.65f, mirror, aspectScale))

            // Upper lash liner
            setMkColor(style.linerColor, style.linerAlpha * 0.24f)
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, LEFT_UPPER_LINER,  leftLinerW  * 0.5f, mirror, aspectScale))
            drawGeometry(MakeupGeometry.buildStrokeMesh(lm, RIGHT_UPPER_LINER, rightLinerW * 0.5f, mirror, aspectScale))

            // Micro-wing: extend slightly past the outer corner tangent.
            fun drawWing(outerIdx: Int, nextIdx: Int, wingLen: Float, wingW: Float) {
                val ox = MakeupGeometry.lmX(lm, outerIdx, mirror, aspectScale)
                val oy = MakeupGeometry.lmY(lm, outerIdx)
                val nx = MakeupGeometry.lmX(lm, nextIdx, mirror, aspectScale)
                val ny = MakeupGeometry.lmY(lm, nextIdx)

                var dx = ox - nx
                var dy = oy - ny
                val d = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat().coerceAtLeast(1e-6f)
                dx /= d; dy /= d

                val wx = ox + dx * wingLen
                val wy = oy + dy * wingLen

                val seg = floatArrayOf(ox, oy, wx, wy)
                drawGeometry(MakeupGeometry.buildStrokeMesh2D(seg, wingW))
            }

            val wingLenL = (leftLinerW * 2.3f).coerceIn(0.008f, 0.045f)
            val wingLenR = (rightLinerW * 2.3f).coerceIn(0.008f, 0.045f)
            val wingWl = (leftLinerW * 0.42f).coerceIn(0.0025f, 0.012f)
            val wingWr = (rightLinerW * 0.42f).coerceIn(0.0025f, 0.012f)

            setMkColor(style.linerColor, style.linerAlpha * 0.26f)
            // Upper liner arrays start at the outer corner on both eyes.
            drawWing(LEFT_UPPER_LINER[0], LEFT_UPPER_LINER[1], wingLenL, wingWl)
            drawWing(RIGHT_UPPER_LINER[0], RIGHT_UPPER_LINER[1], wingLenR, wingWr)
        }

        // ── Lashes (natural mascara) ─────────────────────────────────────────
        if (style.lashAlpha > 0.01f) {
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glUniform1f(mkUBlendMode, 1f)  // multiply-like
            GLES20.glUniform1f(mkUGradientDir, 0f)

            val leftEyeH  = MakeupGeometry.eyeHeight(lm, LEFT_UPPER_LID_ARC,  LandmarkIndex.LEFT_EYE_LOWER,  mirror, aspectScale)
            val rightEyeH = MakeupGeometry.eyeHeight(lm, RIGHT_UPPER_LID_ARC, LandmarkIndex.RIGHT_EYE_LOWER, mirror, aspectScale)

            val leftLen  = (leftEyeH * 0.55f).coerceIn(0.006f, 0.050f)
            val rightLen = (rightEyeH * 0.55f).coerceIn(0.006f, 0.050f)
            val leftTh   = (leftEyeH * 0.055f).coerceIn(0.0012f, 0.010f)
            val rightTh  = (rightEyeH * 0.055f).coerceIn(0.0012f, 0.010f)

            setMkColor(style.lashColor, style.lashAlpha)
            drawGeometry(
                MakeupGeometry.buildUpperLashesMesh(
                    lm = lm,
                    upperLinerIndices = LEFT_UPPER_LINER,
                    isMirrored = mirror,
                    lengthNdc = leftLen,
                    thicknessNdc = leftTh,
                    aspectScale = aspectScale,
                    subdivisionsPerSegment = 4,
                )
            )
            drawGeometry(
                MakeupGeometry.buildUpperLashesMesh(
                    lm = lm,
                    upperLinerIndices = RIGHT_UPPER_LINER,
                    isMirrored = mirror,
                    lengthNdc = rightLen,
                    thicknessNdc = rightTh,
                    aspectScale = aspectScale,
                    subdivisionsPerSegment = 4,
                )
            )
        }

        // ── Lips ──────────────────────────────────────────────────────────────
        if (style.lipAlpha > 0.01f) {
            GLES20.glUniform1f(mkUGradientDir, 0f)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

            // Ring mesh: covers only lip tissue between outer and inner rings.
            // No triangle extends inside LIPS_INNER, so the mouth opening is
            // never painted.
            val lipRing = MakeupGeometry.buildLipRingMesh(lm, LIPS_OUTER, LIPS_INNER, mirror, aspectScale)

            // Layer 1: soft multiply-style base for natural lip texture
            GLES20.glUniform1f(mkUBlendMode, 1f)
            setMkColor(style.lipColor, style.lipAlpha * 0.22f)
            drawGeometry(lipRing)

            // Layer 2: main pigment fill
            setMkColor(style.lipColor, style.lipAlpha * 0.80f)
            drawGeometry(lipRing)

            // Layer 3: specular highlight on cupid's bow (normal blend)
            GLES20.glUniform1f(mkUBlendMode, 0f)
            val highlightColor = lightenColor(style.lipColor, 0.55f)
            setMkColor(highlightColor, style.lipAlpha * 0.22f)
            drawGeometry(MakeupGeometry.buildFanMesh(lm, LIP_CUPID_BOW, mirror, aspectScale))

            // Layer 4: re-saturate with multiply pass on ring only
            GLES20.glBlendFunc(GLES20.GL_DST_COLOR, GLES20.GL_ZERO)
            setMkColor(style.lipColor, style.lipAlpha * 0.28f)
            drawGeometry(lipRing)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        }

        // ── Highlighter (Bridal Glow etc.) ────────────────────────────────────
        if (style.highlightAlpha > 0.01f) {
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE)
            GLES20.glUniform1f(mkUBlendMode, 0f)
            GLES20.glUniform1f(mkUGradientDir, 0f)
            setMkColor(style.highlightColor, style.highlightAlpha * 0.5f)
            drawGeometry(MakeupGeometry.buildFanMesh(lm, NOSE_BRIDGE, mirror, aspectScale))
        }

        GLES20.glDisableVertexAttribArray(mkAPosition)
        GLES20.glDisableVertexAttribArray(mkAEdgeFactor)
        GLES20.glDisableVertexAttribArray(mkARegionUV)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Uploads [verts] to the shared makeup VBO and draws with the foundation
     * shader's attribute locations.  Vertex layout is identical to [drawGeometry]:
     * [ndcX, ndcY, edgeFactor, regionU, regionV].
     */
    private fun drawFoundation(verts: FloatArray) {
        if (verts.isEmpty()) return
        uploadGeometry(verts)

        GLES20.glEnableVertexAttribArray(foundAPosition)
        GLES20.glVertexAttribPointer(foundAPosition, 2, GLES20.GL_FLOAT, false, STRIDE, 0)

        GLES20.glEnableVertexAttribArray(foundAEdgeFactor)
        GLES20.glVertexAttribPointer(foundAEdgeFactor, 1, GLES20.GL_FLOAT, false, STRIDE, 2 * FLOAT_SIZE)

        GLES20.glEnableVertexAttribArray(foundARegionUV)
        GLES20.glVertexAttribPointer(foundARegionUV, 2, GLES20.GL_FLOAT, false, STRIDE, 3 * FLOAT_SIZE)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, verts.size / 5)

        GLES20.glDisableVertexAttribArray(foundAPosition)
        GLES20.glDisableVertexAttribArray(foundAEdgeFactor)
        GLES20.glDisableVertexAttribArray(foundARegionUV)
    }

    private fun drawGeometry(verts: FloatArray) {
        if (verts.isEmpty()) return
        uploadGeometry(verts)

        GLES20.glEnableVertexAttribArray(mkAPosition)
        GLES20.glVertexAttribPointer(mkAPosition, 2, GLES20.GL_FLOAT, false, STRIDE, 0)

        GLES20.glEnableVertexAttribArray(mkAEdgeFactor)
        GLES20.glVertexAttribPointer(mkAEdgeFactor, 1, GLES20.GL_FLOAT, false, STRIDE, 2 * FLOAT_SIZE)

        GLES20.glEnableVertexAttribArray(mkARegionUV)
        GLES20.glVertexAttribPointer(mkARegionUV, 2, GLES20.GL_FLOAT, false, STRIDE, 3 * FLOAT_SIZE)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, verts.size / 5)
    }

    /** Copies [verts] into the pre-allocated staging buffer and uploads to [mkVbo]. */
    private fun uploadGeometry(verts: FloatArray) {
        geomFloatBuffer.clear()
        geomFloatBuffer.put(verts)
        geomBuffer.position(0).limit(verts.size * FLOAT_SIZE)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, mkVbo)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, verts.size * FLOAT_SIZE, geomBuffer, GLES20.GL_DYNAMIC_DRAW)

        geomBuffer.limit(geomBuffer.capacity()) // restore limit for next use
    }

    private fun setMkColor(color: Color, alpha: Float) {
        GLES20.glUniform4f(mkUColor, color.red, color.green, color.blue, alpha.coerceIn(0f, 1f))
        GLES20.glUniform4f(mkUColor2, color.red, color.green, color.blue, alpha.coerceIn(0f, 1f))
    }

    private fun setMkColor4(c1: Color, a1: Float, c2: Color, a2: Float) {
        GLES20.glUniform4f(mkUColor, c1.red, c1.green, c1.blue, a1.coerceIn(0f, 1f))
        GLES20.glUniform4f(mkUColor2, c2.red, c2.green, c2.blue, a2.coerceIn(0f, 1f))
    }

    private fun darkenColor(c: Color, factor: Float) =
        Color(c.red * factor, c.green * factor, c.blue * factor, c.alpha)

    private fun lightenColor(c: Color, factor: Float) =
        Color(
            (c.red + (1f - c.red) * factor).coerceIn(0f, 1f),
            (c.green + (1f - c.green) * factor).coerceIn(0f, 1f),
            (c.blue + (1f - c.blue) * factor).coerceIn(0f, 1f),
            c.alpha,
        )

    private fun createNoiseTexture(size: Int, seed: Int): Int {
        val texIds = IntArray(1)
        GLES20.glGenTextures(1, texIds, 0)
        val id = texIds[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_REPEAT)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_REPEAT)

        // Small grayscale noise with both coarse blotches and fine grain.
        val coarseSize = 8
        val coarse = FloatArray(coarseSize * coarseSize)
        var s = seed
        fun nextFloat(): Float {
            s = (1664525 * s + 1013904223)
            val v = (s ushr 8) and 0x00FFFFFF
            return v.toFloat() / 16777215f
        }
        for (i in coarse.indices) coarse[i] = nextFloat()

        fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t
        fun smooth(t: Float) = t * t * (3f - 2f * t)
        fun sampleCoarse(u: Float, v: Float): Float {
            val x = u * coarseSize
            val y = v * coarseSize
            val x0 = floor(x).toInt().coerceIn(0, coarseSize - 1)
            val y0 = floor(y).toInt().coerceIn(0, coarseSize - 1)
            val x1 = (x0 + 1).coerceIn(0, coarseSize - 1)
            val y1 = (y0 + 1).coerceIn(0, coarseSize - 1)
            val tx = smooth((x - x0.toFloat()).coerceIn(0f, 1f))
            val ty = smooth((y - y0.toFloat()).coerceIn(0f, 1f))
            val a = coarse[y0 * coarseSize + x0]
            val b = coarse[y0 * coarseSize + x1]
            val c = coarse[y1 * coarseSize + x0]
            val d = coarse[y1 * coarseSize + x1]
            return lerp(lerp(a, b, tx), lerp(c, d, tx), ty)
        }

        val gray = ByteArray(size * size)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val u = x.toFloat() / size.toFloat()
                val v = y.toFloat() / size.toFloat()
                val low = sampleCoarse(u, v)
                val high = nextFloat()
                val n = (0.5f + (low - 0.5f) * 0.65f + (high - 0.5f) * 0.20f).coerceIn(0f, 1f)
                gray[y * size + x] = (n * 255f).toInt().coerceIn(0, 255).toByte()
            }
        }

        // Make the tile seamless under GL_REPEAT + GL_LINEAR by matching wrap edges.
        for (x in 0 until size) {
            gray[(size - 1) * size + x] = gray[0 * size + x]
        }
        for (y in 0 until size) {
            gray[y * size + (size - 1)] = gray[y * size + 0]
        }
        gray[(size - 1) * size + (size - 1)] = gray[0]

        val buf = ByteBuffer.allocateDirect(size * size * 4).order(ByteOrder.nativeOrder())
        for (i in 0 until size * size) {
            val g = gray[i]
            buf.put(g); buf.put(g); buf.put(g); buf.put(0xFF.toByte())
        }
        buf.position(0)

        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D,
            0,
            GLES20.GL_RGBA,
            size,
            size,
            0,
            GLES20.GL_RGBA,
            GLES20.GL_UNSIGNED_BYTE,
            buf,
        )
        return id
    }



    // ─────────────────────────────────────────────────────────────────────────
    // Photo capture
    // ─────────────────────────────────────────────────────────────────────────

    /** Request a pixel readback on the next rendered frame. */
    fun requestCapture() { captureNextFrame = true }

    private fun readPixels() {
        val w = viewWidth; val h = viewHeight
        val buf = ByteBuffer.allocateDirect(w * h * 4)
        buf.order(ByteOrder.nativeOrder())
        GLES20.glReadPixels(0, 0, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
        buf.rewind()
        val bytes = ByteArray(w * h * 4)
        buf.get(bytes)
        // GL reads bottom-up, flip rows
        val flipped = flipVertically(bytes, w, h)
        onPixelsReady?.invoke(flipped, w, h)
    }

    private fun flipVertically(src: ByteArray, w: Int, h: Int): ByteArray {
        val dst = ByteArray(src.size)
        val rowSize = w * 4
        for (y in 0 until h) {
            System.arraycopy(src, y * rowSize, dst, (h - 1 - y) * rowSize, rowSize)
        }
        return dst
    }

    /**
     * Runs [block] on the GL thread, catching any exception and logging it
     * instead of letting it propagate to the GL thread where it would be
     * unhandled and kill the whole app.
     */
    private inline fun runGLSafely(tag: String, block: () -> Unit) {
        try {
            block()
        } catch (e: Exception) {
            Log.e(TAG, "GL error in $tag", e)
        }
    }
}
