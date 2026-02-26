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

    enum class PreviewScaleMode {
        /**
         * Preserve aspect ratio and show the entire camera frame.
         * May show letterbox/pillarbox bars when the view aspect differs.
         */
        FIT_CENTER,

        /**
         * Preserve aspect ratio and fill the entire view.
         * Crops only when the view and camera aspect ratios differ.
         */
        FILL_CENTER,
    }

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

    // ── Offscreen base FBO (used only for contour relighting) ────────────────
    private var baseFbo = 0
    private var baseTex = 0
    private var baseTexW = 0
    private var baseTexH = 0

    // ── Lighting metrics (EMA-smoothed) for contour/highlight auto-tuning ────
    private var faceMeanYEma = 0.50f
    private var faceStdYEma = 0.12f
    private var clipFracEma = 0.00f
    private var lightBiasEma = 0.50f
    private var hasLightingEma = false

    // ── Camera shader ─────────────────────────────────────────────────────────
    private var camProg = 0
    private var camAPosition = 0
    private var camATexCoord = 0
    private var camUMirror = 0
    private var camUTone = 0
    private var camUTexture = 0
    private var camUCropScale = 0

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
    private var foundURadiusScale = -1
    private var foundUConcealLift = -1
    private var foundUConcealNeutralize = -1
    private var foundUAutoCorrect = -1
    private var foundUAutoThreshold = -1
    private var foundUAutoRadiusScale = -1
    private var foundUMirror = 0
    private var foundUCropScale = 0

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
    private var mkUTime = -1
    private var mkUCropScale = 0
    private var mkUCameraTex = -1
    private var mkUMirror = -1
    private var mkUSkinLuma = -1

    private var noiseTextureId = 0
    private var startTimeNanos: Long = 0L

    // ── Contour (standalone mask-based effect) ────────────────────────────────
    private val contourRenderer = ContourRenderer(context)
    private val contourUvTransform3x3 = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f,
    )

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

    /**
     * Estimated user skin tint (sRGB 0..1), derived from the *analysis* RGBA frames.
     * Smoothed with an EMA to avoid flicker under noise/motion.
     *
     * Note: landmarks are in analysis-frame coordinates (unmirrored); we sample
     * pixels using the raw landmark x/y and only mirror later for rendering.
     */
    private var skinTintEma: Color? = null

    /** Skin smoothing intensity (0–1). */
    var smoothIntensity = 0.45f
    var toneIntensity = 0.35f

    /**
     * Auto-correct strength (0–1): detects locally-dark patches within the face oval
     * and lifts/neutralizes them. Runs even when the selected style is "None".
     */
    var autoCorrectStrength = 0.55f

    /** Luma-delta threshold for dark-patch detection (typical range ~0.02–0.06). */
    var autoCorrectThreshold = 0.024f

    /** Radius scale for the wide local-average used in detection. */
    var autoCorrectRadiusScale = 1.0f

    /** Whether we're currently mirroring (true for live preview). */
    var isMirrored = true

    /**
     * How the camera background should be scaled into the view.
     *
     * Default is [FIT_CENTER] so the entire camera frame is always visible.
     * This preserves aspect ratio and adds letterbox/pillarbox bars when needed.
     */
    @Volatile var previewScaleMode: PreviewScaleMode = PreviewScaleMode.FIT_CENTER

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

        // Offscreen base texture + FBO (allocated onSurfaceChanged when size known).
        val baseTexIds = IntArray(1)
        GLES20.glGenTextures(1, baseTexIds, 0)
        baseTex = baseTexIds[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, baseTex)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        val fbos = IntArray(1)
        GLES20.glGenFramebuffers(1, fbos, 0)
        baseFbo = fbos[0]

        // Compile shaders
        val camVert = loadRawString(context, R.raw.camera_vertex)
        val camFrag = loadRawString(context, R.raw.camera_fragment)
        camProg = linkProgram(camVert, camFrag)
        camAPosition     = GLES20.glGetAttribLocation(camProg, "aPosition")
        camATexCoord     = GLES20.glGetAttribLocation(camProg, "aTexCoord")
        camUMirror       = GLES20.glGetUniformLocation(camProg, "uMirror")
        camUTone         = GLES20.glGetUniformLocation(camProg, "uTone")
        camUTexture      = GLES20.glGetUniformLocation(camProg, "uTexture")
        camUCropScale    = GLES20.glGetUniformLocation(camProg, "uCropScale")

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
        foundURadiusScale = GLES20.glGetUniformLocation(foundProg, "uRadiusScale")
        foundUConcealLift = GLES20.glGetUniformLocation(foundProg, "uConcealLift")
        foundUConcealNeutralize = GLES20.glGetUniformLocation(foundProg, "uConcealNeutralize")
        foundUAutoCorrect = GLES20.glGetUniformLocation(foundProg, "uAutoCorrect")
        foundUAutoThreshold = GLES20.glGetUniformLocation(foundProg, "uAutoThreshold")
        foundUAutoRadiusScale = GLES20.glGetUniformLocation(foundProg, "uAutoRadiusScale")
        foundUMirror      = GLES20.glGetUniformLocation(foundProg,  "uMirror")
        foundUCropScale   = GLES20.glGetUniformLocation(foundProg,  "uCropScale")

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
        mkUTime = GLES20.glGetUniformLocation(mkProg, "uTime")
        mkUCropScale = GLES20.glGetUniformLocation(mkProg, "uCropScale")
        mkUCameraTex = GLES20.glGetUniformLocation(mkProg, "uCameraTex")
        mkUMirror = GLES20.glGetUniformLocation(mkProg, "uMirror")
        mkUSkinLuma = GLES20.glGetUniformLocation(mkProg, "uSkinLuma")

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
        startTimeNanos = System.nanoTime()

        // Standalone contour effect (mask generation + composite).
        contourRenderer.onSurfaceCreated()
        contourRenderer.setUvTransform3x3(contourUvTransform3x3)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) = runGLSafely("onSurfaceChanged") {
        viewWidth = width
        viewHeight = height
        GLES20.glViewport(0, 0, width, height)
        ensureBaseFbo(width, height)
        contourRenderer.onSurfaceChanged(width, height)
    }

    override fun onDrawFrame(gl: GL10?) = runGLSafely("onDrawFrame") {
        // Consume the latest analysis frame (if any) and upload it to the 2D texture.
        val packet = pendingFrame.getAndSet(null)
        if (packet != null) {
            uploadCameraTexture(packet)
            latestLandmarks.set(packet.landmarks)
            val lm = packet.landmarks
            if (lm != null) {
                updateSkinTintEmaFromFrame(packet.rgba, packet.width, packet.height, lm)
                updateFaceLightingMetricsEmaFromFrame(packet.rgba, packet.width, packet.height, lm)
            }
            releaseRgbaBuffer(packet.rgba)
        }

        val (scaleX, scaleY) = computePreviewScale()
        contourRenderer.setCropScale(scaleX, scaleY)

        val style = currentStyle
        val lm = packet?.landmarks ?: latestLandmarks.get()
        val hasAnyLayer =
            autoCorrectStrength > 0.01f ||
            style.foundationAlpha > 0.01f ||
                style.concealerLift > 0.01f ||
                style.concealerNeutralize > 0.01f ||
                style.contourAlpha > 0.01f ||
                style.blushAlpha > 0.01f ||
                style.eyeshadowAlpha > 0.01f ||
                style.linerAlpha > 0.01f ||
                style.lipAlpha > 0.01f ||
                style.highlightAlpha > 0.01f ||
                style.browAlpha > 0.01f ||
                style.browFillAlpha > 0.01f ||
                style.lashAlpha > 0.01f
        val shouldRelightContourHighlight =
            (lm != null && (style.contourAlpha > 0.01f || style.highlightAlpha > 0.01f))

        if (lm != null && hasAnyLayer) {
            if (shouldRelightContourHighlight && baseFbo != 0 && baseTex != 0 && contourRenderer.isReady()) {
                // Offscreen base (camera + foundation/concealer) → onscreen relighting contour → overlays.
                renderCameraToBaseFbo(scaleX, scaleY)
                drawBaseFacePasses(lm, style, scaleX, scaleY)

                // Now draw relit base to the screen.
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
                GLES20.glViewport(0, 0, viewWidth, viewHeight)
                // Do NOT clear here: the contour relight pass overwrites the full frame.
                // Clearing first risks a black preview if contour fails (e.g., shader compile/link issue).
                val ok = drawContourRelightToScreen(lm, style, scaleX, scaleY)
                if (ok) {
                    drawMakeupOverlays(lm, style, scaleX, scaleY, allowLegacyHighlight = false)
                } else {
                    // Fail-safe fallback: render the normal pipeline to screen.
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                    renderCameraToScreen(scaleX, scaleY)
                    drawBaseFacePasses(lm, style, scaleX, scaleY)
                    drawMakeupOverlays(lm, style, scaleX, scaleY, allowLegacyHighlight = false)
                }
            } else {
                // Standard pipeline: camera + base passes + overlays directly to screen.
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
                GLES20.glViewport(0, 0, viewWidth, viewHeight)
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                renderCameraToScreen(scaleX, scaleY)
                drawBaseFacePasses(lm, style, scaleX, scaleY)
                drawMakeupOverlays(lm, style, scaleX, scaleY, allowLegacyHighlight = false)
            }
        } else {
            // No landmarks or no active layers: just draw the camera frame.
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
            GLES20.glViewport(0, 0, viewWidth, viewHeight)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            renderCameraToScreen(scaleX, scaleY)
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

    private fun ensureBaseFbo(w: Int, h: Int) {
        val ww = w.coerceAtLeast(1)
        val hh = h.coerceAtLeast(1)
        if (baseTex == 0 || baseFbo == 0) return
        if (ww == baseTexW && hh == baseTexH) return
        baseTexW = ww
        baseTexH = hh

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, baseTex)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D,
            0,
            GLES20.GL_RGBA,
            baseTexW,
            baseTexH,
            0,
            GLES20.GL_RGBA,
            GLES20.GL_UNSIGNED_BYTE,
            null,
        )
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, baseFbo)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            baseTex,
            0,
        )
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
    }

    private fun renderCameraToBaseFbo(scaleX: Float, scaleY: Float) {
        if (camTexWidth <= 0 || camTexHeight <= 0 || baseFbo == 0 || baseTex == 0) return
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, baseFbo)
        GLES20.glViewport(0, 0, viewWidth, viewHeight)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        GLES20.glDisable(GLES20.GL_BLEND)

        GLES20.glUseProgram(camProg)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)

        GLES20.glUniform1i(camUTexture, 0)
        GLES20.glUniform1f(camUMirror, if (isMirrored) 1f else 0f)
        GLES20.glUniform1f(camUTone, toneIntensity)
        GLES20.glUniform2f(camUCropScale, scaleX, scaleY)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        val stride4 = 4 * FLOAT_SIZE
        GLES20.glEnableVertexAttribArray(camAPosition)
        GLES20.glVertexAttribPointer(camAPosition, 2, GLES20.GL_FLOAT, false, stride4, 0)
        GLES20.glEnableVertexAttribArray(camATexCoord)
        GLES20.glVertexAttribPointer(camATexCoord, 2, GLES20.GL_FLOAT, false, stride4, 2 * FLOAT_SIZE)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }

    private fun renderCameraToScreen(scaleX: Float, scaleY: Float) {
        if (camTexWidth <= 0 || camTexHeight <= 0) return
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(camProg)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)

        GLES20.glUniform1i(camUTexture, 0)
        GLES20.glUniform1f(camUMirror, if (isMirrored) 1f else 0f)
        GLES20.glUniform1f(camUTone, toneIntensity)
        GLES20.glUniform2f(camUCropScale, scaleX, scaleY)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        val stride4 = 4 * FLOAT_SIZE
        GLES20.glEnableVertexAttribArray(camAPosition)
        GLES20.glVertexAttribPointer(camAPosition, 2, GLES20.GL_FLOAT, false, stride4, 0)
        GLES20.glEnableVertexAttribArray(camATexCoord)
        GLES20.glVertexAttribPointer(camATexCoord, 2, GLES20.GL_FLOAT, false, stride4, 2 * FLOAT_SIZE)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }

    /**
     * Compute clip-space scale (sx, sy) that preserves camera aspect ratio while
     * mapping into the current view according to [previewScaleMode].
     *
     * Implementation detail:
     * - Scaling < 1.0 on one axis produces letterbox/pillarbox (FIT_CENTER)
     * - Scaling > 1.0 on one axis produces center-crop (FILL_CENTER)
     */
    private fun computePreviewScale(): Pair<Float, Float> {
        val vw = viewWidth
        val vh = viewHeight
        val tw = camTexWidth
        val th = camTexHeight
        if (vw <= 0 || vh <= 0 || tw <= 0 || th <= 0) return Pair(1f, 1f)

        val viewAspect = vw.toFloat() / vh.toFloat()
        val texAspect = tw.toFloat() / th.toFloat()
        if (viewAspect <= 0f || texAspect <= 0f) return Pair(1f, 1f)

        return when (previewScaleMode) {
            PreviewScaleMode.FIT_CENTER -> {
                if (texAspect > viewAspect) {
                    // Texture wider than view: fit height, shrink Y (bars top/bottom).
                    Pair(1f, viewAspect / texAspect)
                } else {
                    // Texture taller than view: fit width, shrink X (bars left/right).
                    Pair(texAspect / viewAspect, 1f)
                }
            }
            PreviewScaleMode.FILL_CENTER -> {
                if (texAspect > viewAspect) {
                    // Texture wider than view: fill height, expand X (crop left/right).
                    Pair(texAspect / viewAspect, 1f)
                } else {
                    // Texture taller than view: fill width, expand Y (crop top/bottom).
                    Pair(1f, viewAspect / texAspect)
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Makeup drawing
    // ─────────────────────────────────────────────────────────────────────────

    private fun drawBaseFacePasses(lm: FloatArray, style: MakeupStyle, cropScaleX: Float, cropScaleY: Float) {
        GLES20.glEnable(GLES20.GL_BLEND)

        val mirror = isMirrored

        // After removing the ViewPort both the camera Preview OES texture and the
        // ImageAnalysis bitmap cover the same full native sensor frame.  The camera
        // stretches that frame to fill the viewport, and the landmark [0,1]→[-1,1]
        // NDC mapping stretches in exactly the same proportion, so no x-axis
        // correction is needed.  aspectScale is kept at 1f (no-op) to preserve the
        // parameter plumbing in MakeupGeometry for potential future use.
        val aspectScale = 1f
        val effTint = effectiveFoundationTint(style)

        // ── Foundation ────────────────────────────────────────────────────────
        // Drawn first so all subsequent makeup layers composite on top.
        // Samples the camera texture within the face-oval mesh boundary,
        // applying skin-smoothing and warmth correction only where the face is.
        val shouldDrawFoundation =
            style.foundationAlpha > 0.01f || autoCorrectStrength > 0.01f
        if (shouldDrawFoundation) {
            GLES20.glUseProgram(foundProg)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

            GLES20.glUniform1i(foundUTexture, 0)
            val texW = if (camTexWidth > 0) camTexWidth else viewWidth
            val texH = if (camTexHeight > 0) camTexHeight else viewHeight
            GLES20.glUniform2f(foundUTexelSize, 1f / texW.toFloat(), 1f / texH.toFloat())
            // Split auto-correct from foundation makeup:
            // when foundationAlpha is 0, we still run dark-patch correction but avoid
            // re-applying tone/smoothing (prevents “double warmth” and unwanted blur).
            val foundationEnabled = style.foundationAlpha > 0.01f
            GLES20.glUniform1f(foundUSmooth, if (foundationEnabled) smoothIntensity else 0f)
            GLES20.glUniform1f(foundUTone, if (foundationEnabled) toneIntensity else 0f)
            GLES20.glUniform1f(foundUFoundAlpha, style.foundationAlpha)
            if (foundUAutoCorrect >= 0) {
                GLES20.glUniform1f(foundUAutoCorrect, autoCorrectStrength.coerceIn(0f, 1f))
            }
            if (foundUAutoThreshold >= 0) {
                GLES20.glUniform1f(foundUAutoThreshold, autoCorrectThreshold.coerceIn(0f, 0.20f))
            }
            if (foundUAutoRadiusScale >= 0) {
                GLES20.glUniform1f(foundUAutoRadiusScale, autoCorrectRadiusScale.coerceIn(0.50f, 2.75f))
            }
            GLES20.glUniform1f(foundUMirror, if (mirror) 1f else 0f)
            GLES20.glUniform2f(foundUCropScale, cropScaleX, cropScaleY)
            if (foundURadiusScale >= 0) GLES20.glUniform1f(foundURadiusScale, 1.0f)
            if (foundUConcealLift >= 0) GLES20.glUniform1f(foundUConcealLift, 0.0f)
            if (foundUConcealNeutralize >= 0) GLES20.glUniform1f(foundUConcealNeutralize, 0.0f)

            if (foundUFoundTint >= 0) {
                GLES20.glUniform3f(foundUFoundTint, effTint.red, effTint.green, effTint.blue)
            }
            if (foundUFoundCoverage >= 0) {
                GLES20.glUniform1f(foundUFoundCoverage, style.foundationCoverage.coerceIn(0f, 1f))
            }

            drawFoundation(MakeupGeometry.buildFanMesh(lm, FACE_OVAL, mirror, aspectScale))
        }

        // ── Concealer (under-eye) ────────────────────────────────────────────
        // Rendered as a localized foundation-shader pass so it samples the camera
        // texture and can lift/neutralize shadow/cast without looking like paint.
        if (style.concealerLift > 0.01f || style.concealerNeutralize > 0.01f) {
            GLES20.glUseProgram(foundProg)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

            GLES20.glUniform1i(foundUTexture, 0)
            val texW = if (camTexWidth > 0) camTexWidth else viewWidth
            val texH = if (camTexHeight > 0) camTexHeight else viewHeight
            GLES20.glUniform2f(foundUTexelSize, 1f / texW.toFloat(), 1f / texH.toFloat())

            // Concealer-only: avoid re-running full foundation smoothing/tone or global auto-correct.
            GLES20.glUniform1f(foundUSmooth, 0f)
            GLES20.glUniform1f(foundUTone, 0f)
            GLES20.glUniform1f(foundUFoundAlpha, 0f)
            if (foundUFoundCoverage >= 0) {
                GLES20.glUniform1f(foundUFoundCoverage, 0f)
            }
            if (foundUAutoCorrect >= 0) {
                GLES20.glUniform1f(foundUAutoCorrect, 0f)
            }

            GLES20.glUniform1f(foundUMirror, if (mirror) 1f else 0f)
            GLES20.glUniform2f(foundUCropScale, cropScaleX, cropScaleY)
            if (foundURadiusScale >= 0) GLES20.glUniform1f(foundURadiusScale, 1.0f)
            if (foundUFoundTint >= 0) {
                GLES20.glUniform3f(foundUFoundTint, effTint.red, effTint.green, effTint.blue)
            }
            if (foundUConcealLift >= 0) {
                GLES20.glUniform1f(foundUConcealLift, style.concealerLift.coerceIn(0f, 1f))
            }
            if (foundUConcealNeutralize >= 0) {
                GLES20.glUniform1f(foundUConcealNeutralize, style.concealerNeutralize.coerceIn(0f, 1f))
            }

            drawFoundation(
                MakeupGeometry.buildUnderEyeConcealerMesh(
                    lm = lm,
                    innerCornerIdx = LandmarkIndex.LEFT_EYE_INNER,
                    outerCornerIdx = LandmarkIndex.LEFT_EYE_OUTER,
                    lowerLidIdx = LandmarkIndex.LEFT_EYE_LOWER,
                    isMirrored = mirror,
                    aspectScale = aspectScale,
                )
            )
            drawFoundation(
                MakeupGeometry.buildUnderEyeConcealerMesh(
                    lm = lm,
                    innerCornerIdx = LandmarkIndex.RIGHT_EYE_INNER,
                    outerCornerIdx = LandmarkIndex.RIGHT_EYE_OUTER,
                    lowerLidIdx = LandmarkIndex.RIGHT_EYE_LOWER,
                    isMirrored = mirror,
                    aspectScale = aspectScale,
                )
            )
        }
    }

    private fun drawContourRelightToScreen(lm: FloatArray, style: MakeupStyle, cropScaleX: Float, cropScaleY: Float): Boolean {
        if (style.contourAlpha <= 0.01f && style.highlightAlpha <= 0.01f) return false
        if (!contourRenderer.isReady()) return false

        val skinBase = skinTintEma ?: effectiveFoundationTint(style)
        val nowNs = System.nanoTime()

        val texW = baseTexW.coerceAtLeast(1)
        val texH = baseTexH.coerceAtLeast(1)

        val ok = contourRenderer.renderToScreen(
            lm = lm,
            isMirrored = isMirrored,
            style = style,
            skinBase = skinBase,
            frameTexture = baseTex,
            frameTexelSizeX = 1f / texW.toFloat(),
            frameTexelSizeY = 1f / texH.toFloat(),
            lighting = ContourRenderer.LightingMetrics(
                faceMeanY = faceMeanYEma,
                faceStdY = faceStdYEma,
                clipFrac = clipFracEma,
                lightBias = lightBiasEma,
            ),
            nowNs = nowNs,
        )

        // Restore default blending for subsequent layers.
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        return ok
    }

    private fun drawMakeupOverlays(
        lm: FloatArray,
        style: MakeupStyle,
        cropScaleX: Float,
        cropScaleY: Float,
        allowLegacyHighlight: Boolean,
    ) {
        val mirror = isMirrored
        val aspectScale = 1f
        val effTint = effectiveFoundationTint(style)

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glUseProgram(mkProg)
        GLES20.glUniform2f(mkUCropScale, cropScaleX, cropScaleY)

        // Bind the camera texture for per-pixel adaptation in the makeup shader
        // (contour uses this to suppress beard/cast-shadow artefacts).
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTextureId)
        if (mkUCameraTex >= 0) GLES20.glUniform1i(mkUCameraTex, 0)
        if (mkUMirror >= 0) GLES20.glUniform1f(mkUMirror, if (mirror) 1f else 0f)
        if (mkUSkinLuma >= 0) {
            val skinLuma = (effTint.red * 0.2126f + effTint.green * 0.7152f + effTint.blue * 0.0722f)
                .coerceIn(0f, 1f)
            GLES20.glUniform1f(mkUSkinLuma, skinLuma)
        }

        if (mkUTime >= 0) {
            val t = (System.nanoTime() - startTimeNanos).toFloat() / 1_000_000_000f
            GLES20.glUniform1f(mkUTime, t)
        }

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
                val browInner = desaturateColor(lightenColor(style.browColor, 0.10f), 0.08f)
                val browTail = darkenColor(style.browColor, 0.82f)
                // Front lighter → tail deeper reads more like real brow work.
                GLES20.glUniform1f(mkUGradientDir, 1f)
                setMkColor4(browInner, style.browFillAlpha * 0.92f, browTail, style.browFillAlpha)
                // Use a ribbon mesh between the two eyebrow contours to prevent
                // triangle fan spill below the underside edge.
                drawGeometry(MakeupGeometry.buildBrowRibbonMesh(lm, LEFT_BROW, mirror, aspectScale))
                drawGeometry(MakeupGeometry.buildBrowRibbonMesh(lm, RIGHT_BROW, mirror, aspectScale))
                GLES20.glUniform1f(mkUGradientDir, 0f)
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

            // Temple anchors (face oval) to push blush toward the hairline like pro placement.
            val lTx = MakeupGeometry.lmX(lm, 127, mirror, aspectScale)
            val lTy = MakeupGeometry.lmY(lm, 127)
            val rTx = MakeupGeometry.lmX(lm, 356, mirror, aspectScale)
            val rTy = MakeupGeometry.lmY(lm, 356)

            val noseX = MakeupGeometry.lmX(lm, LandmarkIndex.NOSE_TIP, mirror, aspectScale)
            val noseY = MakeupGeometry.lmY(lm, LandmarkIndex.NOSE_TIP)

            // Outer-eye span is a good face-scale proxy (stable across expressions).
            val outerEyeSpan = sqrt((lEx - rEx) * (lEx - rEx) + (lEy - rEy) * (lEy - rEy))
                .coerceIn(0.10f, 1.20f)

            // Cheekbone ridge anchors (higher on the face than "cheek center").
            // These indices come from the cheek region arrays and sit closer to the zygomatic bone.
            val leftCheekboneIdx = intArrayOf(123, 116, 117, 118, 101)
            val rightCheekboneIdx = intArrayOf(352, 346, 347, 348, 330)

            data class BlushPose(
                val cx: Float,
                val cy: Float,
                val axisXx: Float,
                val axisXy: Float,
                val axisYx: Float,
                val axisYy: Float,
                val base: Float,
            )

            fun lerp(a: Float, b: Float, t0: Float): Float {
                val t = t0.coerceIn(0f, 1f)
                return a + (b - a) * t
            }

            fun avgPoint(indices: IntArray): Pair<Float, Float> {
                var sx = 0f
                var sy = 0f
                val n = indices.size.coerceAtLeast(1)
                for (idx in indices) {
                    sx += MakeupGeometry.lmX(lm, idx, mirror, aspectScale)
                    sy += MakeupGeometry.lmY(lm, idx)
                }
                return Pair(sx / n.toFloat(), sy / n.toFloat())
            }

            fun blushPose(
                cx0: Float,
                cy0: Float,
                ex: Float,
                ey: Float,
                mx: Float,
                my: Float,
                tx: Float,
                ty: Float,
                rx: Float,
                ry: Float,
            ): BlushPose {
                // Cheekbone axis: mouth corner → temple (sweeps toward hairline like pro placement).
                // We also compute eye↔mouth span for conservative clamps.
                val eyeMouthSpan = sqrt((ex - mx) * (ex - mx) + (ey - my) * (ey - my)).coerceAtLeast(1e-6f)
                var vx = tx - mx
                var vy = ty - my
                val mtLen = sqrt(vx * vx + vy * vy).coerceAtLeast(1e-6f)
                vx /= mtLen; vy /= mtLen

                // Perpendicular "up" (positive NDC y).
                var upX = -vy
                var upY = vx
                if (upY < 0f) { upX = -upX; upY = -upY }

                val base = min(rx, ry).coerceAtLeast(1e-4f)

                // Professional placement (rule-of-thumb encoded with landmarks):
                // - center should live in the upper-outer cheek quadrant
                // - sweep toward the temple, but not all the way back to the hairline
                // - sit above the cheek hollow (avoid drifting toward the mouth/jaw)
                val along = (mtLen * 0.52f).coerceIn(eyeMouthSpan * 0.42f, eyeMouthSpan * 0.78f)
                val lift = (eyeMouthSpan * 0.16f).coerceIn(0.010f, 0.095f)
                var targetCx = mx + vx * along + upX * lift
                var targetCy = my + vy * along + upY * lift

                // Blend with the cheekbone ridge centroid so placement stays on-bone across face shapes.
                val blendT = 0.68f
                var cx = lerp(cx0, targetCx, blendT)
                var cy = lerp(cy0, targetCy, blendT)

                // Safety clamp: keep blush below the outer eye by a small margin (avoids under-eye placement).
                // NDC y increases upward, so "below" means <= (eyeY - margin).
                val eyeMargin = (eyeMouthSpan * 0.11f).coerceIn(0.010f, 0.070f)
                cy = min(cy, ey - eyeMargin)

                // Keep blush off the "apple" / lower cheek: ensure it stays above mouth corner by a margin.
                val mouthMarginUp = (eyeMouthSpan * 0.18f).coerceIn(0.012f, 0.090f)
                cy = maxOf(cy, my + mouthMarginUp)

                // Keep blush away from the nose (classic "two-finger widths from nose" rule).
                // Push outward along nose→temple direction if it drifts too medial.
                var outX = tx - noseX
                var outY = ty - noseY
                val outLen = sqrt(outX * outX + outY * outY).coerceAtLeast(1e-6f)
                outX /= outLen; outY /= outLen
                val minOut = (outerEyeSpan * 0.34f).coerceIn(0.035f, 0.26f)
                val outProj = (cx - noseX) * outX + (cy - noseY) * outY
                if (outProj < minOut) {
                    val push = (minOut - outProj)
                    cx += outX * push
                    cy += outY * push * 0.35f // slightly bias upward with the push
                }

                return BlushPose(
                    cx = cx,
                    cy = cy,
                    axisXx = vx,
                    axisXy = vy,
                    axisYx = upX,
                    axisYy = upY,
                    base = base,
                )
            }

            val (lBx, lBy) = avgPoint(leftCheekboneIdx)
            val (rBx, rBy) = avgPoint(rightCheekboneIdx)

            // Fall back to cheek-center if the cheekbone centroid ever degenerates (shouldn't, but safe).
            val lPose = blushPose(
                cx0 = if (lBx.isFinite()) lBx else lCx0,
                cy0 = if (lBy.isFinite()) lBy else lCy0,
                ex = lEx,
                ey = lEy,
                mx = lMx,
                my = lMy,
                tx = lTx,
                ty = lTy,
                rx = lRx,
                ry = lRy,
            )
            val rPose = blushPose(
                cx0 = if (rBx.isFinite()) rBx else rCx0,
                cy0 = if (rBy.isFinite()) rBy else rCy0,
                ex = rEx,
                ey = rEy,
                mx = rMx,
                my = rMy,
                tx = rTx,
                ty = rTy,
                rx = rRx,
                ry = rRy,
            )

            GLES20.glUniform1f(mkUBlendMode, 0f)
            GLES20.glUniform1f(mkUGradientDir, 0f)
            GLES20.glUniform1f(mkUEffectKind, 1f)
            // Lower noise: the shader uses it mostly as micro-dither now (cleaner, more pro finish).
            GLES20.glUniform1f(mkUNoiseAmount, 0.085f)

            fun drawBlush(p: BlushPose, radiusScale: Float, alphaScale: Float) {
                // More "swept" shape: longer along cheekbone, tighter vertically.
                val radX = (p.base * 2.05f * radiusScale).coerceIn(0.012f, 0.48f)
                val radY = (p.base * 0.96f * radiusScale).coerceIn(0.008f, 0.30f)
                setMkColor(style.blushColor, (style.blushAlpha * alphaScale).coerceIn(0f, 0.90f))
                drawGeometry(
                    MakeupGeometry.buildRotatedEllipseMesh(
                        centerX = p.cx,
                        centerY = p.cy,
                        radiusX = radX,
                        radiusY = radY,
                        axisXx = p.axisXx,
                        axisXy = p.axisXy,
                        axisYx = p.axisYx,
                        axisYy = p.axisYy,
                        segments = 46,
                    )
                )
            }

            // Main pass (soft-light shader path is more subtle than flat paint, so keep alpha modestly higher).
            drawBlush(lPose, radiusScale = 1.00f, alphaScale = 0.74f)
            drawBlush(rPose, radiusScale = 1.00f, alphaScale = 0.74f)

            // Second softer pass for depth
            drawBlush(lPose, radiusScale = 1.48f, alphaScale = 0.34f)
            drawBlush(rPose, radiusScale = 1.48f, alphaScale = 0.34f)

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

            // Sparkle overlay (twinkling glitter) — additive so it reads as shimmer.
            if (style.eyeshadowSparkle > 0.01f) {
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE)
                GLES20.glUniform1f(mkUBlendMode, 0f)
                GLES20.glUniform1f(mkUGradientDir, 0f)
                GLES20.glUniform1f(mkUEffectKind, 2f)
                GLES20.glUniform1f(mkUNoiseScale, 58.0f)
                GLES20.glUniform1f(mkUNoiseAmount, style.eyeshadowSparkle.coerceIn(0f, 1f))

                val pearl = lightenColor(style.eyeshadowColor, 0.86f)
                // Slight boost so sparkles read on typical phone exposure.
                setMkColor(pearl, (0.38f * style.eyeshadowSparkle).coerceIn(0f, 0.45f))
                drawGeometry(leftMesh)
                drawGeometry(rightMesh)

                // Restore defaults for subsequent layers.
                GLES20.glUniform1f(mkUEffectKind, 0f)
                GLES20.glUniform1f(mkUNoiseAmount, 0f)
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
                GLES20.glUniform1f(mkUGradientDir, 1f)
            }
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

            val leftLen  = (leftEyeH * 0.754f).coerceIn(0.006f, 0.052f)
            val rightLen = (rightEyeH * 0.754f).coerceIn(0.006f, 0.052f)
            val leftTh   = (leftEyeH * 0.095f).coerceIn(0.0014f, 0.0130f)
            val rightTh  = (rightEyeH * 0.095f).coerceIn(0.0014f, 0.0130f)

            val leftLineW  = (leftEyeH * 0.040f).coerceIn(0.0011f, 0.0065f)
            val rightLineW = (rightEyeH * 0.040f).coerceIn(0.0011f, 0.0065f)

            fun drawUpperLashes(upper: IntArray, len: Float, th: Float, lineW: Float) {
                // Lash-line density: fills gaps between spikes for a more pro finish.
                setMkColor(style.lashColor, style.lashAlpha * 0.18f)
                drawGeometry(MakeupGeometry.buildStrokeMesh(lm, upper, lineW * 1.55f, mirror, aspectScale))
                setMkColor(style.lashColor, style.lashAlpha * 0.26f)
                drawGeometry(MakeupGeometry.buildStrokeMesh(lm, upper, lineW * 1.05f, mirror, aspectScale))

                // Two-pass spikes: base volume + defined tips.
                val subs = 5

                setMkColor(style.lashColor, style.lashAlpha * 0.42f)
                drawGeometry(
                    MakeupGeometry.buildUpperLashesMesh(
                        lm = lm,
                        upperLinerIndices = upper,
                        isMirrored = mirror,
                        lengthNdc = (len * 0.92f).coerceIn(0.006f, 0.060f),
                        thicknessNdc = (th * 1.70f).coerceIn(0.0012f, 0.020f),
                        aspectScale = aspectScale,
                        subdivisionsPerSegment = subs,
                    )
                )

                setMkColor(style.lashColor, style.lashAlpha * 0.68f)
                drawGeometry(
                    MakeupGeometry.buildUpperLashesMesh(
                        lm = lm,
                        upperLinerIndices = upper,
                        isMirrored = mirror,
                        lengthNdc = len,
                        thicknessNdc = (th * 1.25f).coerceIn(0.0012f, 0.020f),
                        aspectScale = aspectScale,
                        subdivisionsPerSegment = subs,
                    )
                )
            }

            drawUpperLashes(LEFT_UPPER_LINER, leftLen, leftTh, leftLineW)
            drawUpperLashes(RIGHT_UPPER_LINER, rightLen, rightTh, rightLineW)
        }

        // ── Lips ──────────────────────────────────────────────────────────────
        if (style.lipAlpha > 0.01f) {
            GLES20.glUniform1f(mkUGradientDir, 0f)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

            // Ring mesh: covers only lip tissue between outer and inner rings.
            // No triangle extends inside LIPS_INNER, so the mouth opening is
            // never painted.
            val lipRing = MakeupGeometry.buildLipRingMesh(
                lm = lm,
                outerIndices = LIPS_OUTER,
                innerIndices = LIPS_INNER,
                isMirrored = mirror,
                innerEdgeFactor = 0.60f,
                aspectScale = aspectScale,
            )

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
        if (allowLegacyHighlight && style.highlightAlpha > 0.01f) {
            // Highlight should melt into the base rather than reading as an
            // additive "sticker" layer. Keep sparkle additive, but blend the
            // highlight body normally.
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glUniform1f(mkUBlendMode, 0f)
            GLES20.glUniform1f(mkUGradientDir, 0f)

            // Dedicated highlight shader mode: softer tail + micro-dither to avoid visible edges.
            GLES20.glUniform1f(mkUEffectKind, 4f)
            GLES20.glUniform1f(mkUNoiseScale, 48.0f)
            GLES20.glUniform1f(mkUNoiseAmount, 0.06f)

            // Inner-eye highlight (tear-duct): small soft glow, scaled by eye size.
            val noseX = MakeupGeometry.lmX(lm, LandmarkIndex.NOSE_TIP, mirror, aspectScale)
            val noseY = MakeupGeometry.lmY(lm, LandmarkIndex.NOSE_TIP)

            // Skin-aware highlight tone:
            // Pure near-white additive highlight can blow out; bias slightly toward the user's base tint.
            fun mixColor(a: Color, b: Color, t0: Float): Color {
                val t = t0.coerceIn(0f, 1f)
                return Color(
                    red   = (a.red   + (b.red   - a.red)   * t).coerceIn(0f, 1f),
                    green = (a.green + (b.green - a.green) * t).coerceIn(0f, 1f),
                    blue  = (a.blue  + (b.blue  - a.blue)  * t).coerceIn(0f, 1f),
                    alpha = 1f,
                )
            }

            val tintSoft = desaturateColor(effTint, 0.35f)
            val hlColor = desaturateColor(mixColor(style.highlightColor, tintSoft, 0.24f), 0.12f)

            fun drawInnerEyeHighlight(
                innerIdx: Int,
                outerIdx: Int,
                upperArc: IntArray,
                lowerIdx: Int,
                color: Color,
                alpha: Float,
                doBloom: Boolean,
            ) {
                val ix0 = MakeupGeometry.lmX(lm, innerIdx, mirror, aspectScale)
                val iy0 = MakeupGeometry.lmY(lm, innerIdx)
                val ox0 = MakeupGeometry.lmX(lm, outerIdx, mirror, aspectScale)
                val oy0 = MakeupGeometry.lmY(lm, outerIdx)

                var ax = ox0 - ix0
                var ay = oy0 - iy0
                val aLen = sqrt(ax * ax + ay * ay).coerceAtLeast(1e-6f)
                ax /= aLen; ay /= aLen

                // Perpendicular "up" (positive NDC y).
                var upX = -ay
                var upY = ax
                if (upY < 0f) { upX = -upX; upY = -upY }

                // Offset toward nose so we land on skin, not the eyeball.
                var nx = noseX - ix0
                var ny = noseY - iy0
                val nLen = sqrt(nx * nx + ny * ny).coerceAtLeast(1e-6f)
                nx /= nLen; ny /= nLen

                val eyeH = MakeupGeometry.eyeHeight(lm, upperArc, lowerIdx, mirror, aspectScale)
                val base = eyeH.coerceIn(0.010f, 0.140f)

                val centerX = ix0 + nx * (base * 0.17f) + upX * (base * 0.09f)
                val centerY = iy0 + ny * (base * 0.17f) + upY * (base * 0.09f)

                val rx = (base * 0.28f).coerceIn(0.0035f, 0.050f)
                val ry = (base * 0.18f).coerceIn(0.0028f, 0.038f)

                if (doBloom) {
                    // Core
                    setMkColor(color, alpha * 0.28f)
                    drawGeometry(MakeupGeometry.buildBlushMesh(centerX, centerY, rx, ry, segments = 38))
                    // Bloom
                    setMkColor(color, alpha * 0.11f)
                    drawGeometry(MakeupGeometry.buildBlushMesh(centerX, centerY, rx * 1.45f, ry * 1.45f, segments = 38))
                } else {
                    // Sparkle pass: single tighter draw.
                    setMkColor(color, alpha)
                    drawGeometry(MakeupGeometry.buildBlushMesh(centerX, centerY, rx, ry, segments = 38))
                }
            }

            drawInnerEyeHighlight(
                innerIdx = LandmarkIndex.LEFT_EYE_INNER,
                outerIdx = LandmarkIndex.LEFT_EYE_OUTER,
                upperArc = LEFT_UPPER_LID_ARC,
                lowerIdx = LandmarkIndex.LEFT_EYE_LOWER,
                color = hlColor,
                alpha = style.highlightAlpha,
                doBloom = true,
            )
            drawInnerEyeHighlight(
                innerIdx = LandmarkIndex.RIGHT_EYE_INNER,
                outerIdx = LandmarkIndex.RIGHT_EYE_OUTER,
                upperArc = RIGHT_UPPER_LID_ARC,
                lowerIdx = LandmarkIndex.RIGHT_EYE_LOWER,
                color = hlColor,
                alpha = style.highlightAlpha,
                doBloom = true,
            )

            // Nose bridge highlight (boosted vs previous single pass).
            val noseFan = MakeupGeometry.buildFanMesh(lm, NOSE_BRIDGE, mirror, aspectScale)
            // Keep as a subtle base; the stripe carries the “pro” specular read.
            setMkColor(hlColor, style.highlightAlpha * 0.10f)
            drawGeometry(noseFan)

            // Sparkle overlay on highlight regions (additive twinkle).
            if (style.highlightSparkle > 0.01f) {
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE)
                GLES20.glUniform1f(mkUEffectKind, 2f)
                GLES20.glUniform1f(mkUNoiseScale, 84.0f)
                GLES20.glUniform1f(mkUNoiseAmount, style.highlightSparkle.coerceIn(0f, 1f))

                val sparkleColor = lightenColor(hlColor, 0.78f)
                // Slight boost so sparkle is visible even when highlightAlpha is modest.
                val a = ((0.05f + style.highlightAlpha * 0.72f) * style.highlightSparkle).coerceIn(0f, 0.28f)

                // Re-draw same highlight meshes with sparkle effect.
                drawInnerEyeHighlight(
                    innerIdx = LandmarkIndex.LEFT_EYE_INNER,
                    outerIdx = LandmarkIndex.LEFT_EYE_OUTER,
                    upperArc = LEFT_UPPER_LID_ARC,
                    lowerIdx = LandmarkIndex.LEFT_EYE_LOWER,
                    color = sparkleColor,
                    alpha = a,
                    doBloom = false,
                )
                drawInnerEyeHighlight(
                    innerIdx = LandmarkIndex.RIGHT_EYE_INNER,
                    outerIdx = LandmarkIndex.RIGHT_EYE_OUTER,
                    upperArc = RIGHT_UPPER_LID_ARC,
                    lowerIdx = LandmarkIndex.RIGHT_EYE_LOWER,
                    color = sparkleColor,
                    alpha = a,
                    doBloom = false,
                )

                // Restore defaults.
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
                GLES20.glUniform1f(mkUEffectKind, 4f)
                GLES20.glUniform1f(mkUNoiseScale, 48.0f)
                GLES20.glUniform1f(mkUNoiseAmount, 0.06f)
            }

            // Restore defaults for any later layers.
            GLES20.glUniform1f(mkUEffectKind, 0f)
            GLES20.glUniform1f(mkUNoiseAmount, 0f)
        }

        GLES20.glDisableVertexAttribArray(mkAPosition)
        GLES20.glDisableVertexAttribArray(mkAEdgeFactor)
        GLES20.glDisableVertexAttribArray(mkARegionUV)
    }

    private fun updateFaceLightingMetricsEmaFromFrame(rgba: ByteBuffer, width: Int, height: Int, lm: FloatArray) {
        if (width <= 8 || height <= 8) return
        if (lm.isEmpty()) return

        // Face ROI bbox from FACE_OVAL landmarks (analysis-frame UV, unmirrored).
        var minX = 1f
        var maxX = 0f
        var minY = 1f
        var maxY = 0f
        for (idx in com.ham.app.data.FACE_OVAL) {
            val x = lm[idx * 3].coerceIn(0f, 1f)
            val y = lm[idx * 3 + 1].coerceIn(0f, 1f)
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }
        val padX = ((maxX - minX) * 0.06f).coerceIn(0.0f, 0.06f)
        val padY = ((maxY - minY) * 0.06f).coerceIn(0.0f, 0.06f)
        minX = (minX - padX).coerceIn(0f, 1f)
        maxX = (maxX + padX).coerceIn(0f, 1f)
        minY = (minY - padY).coerceIn(0f, 1f)
        maxY = (maxY + padY).coerceIn(0f, 1f)

        val x0 = (minX * (width - 1)).toInt().coerceIn(0, width - 1)
        val x1 = (maxX * (width - 1)).toInt().coerceIn(0, width - 1)
        val y0 = (minY * (height - 1)).toInt().coerceIn(0, height - 1)
        val y1 = (maxY * (height - 1)).toInt().coerceIn(0, height - 1)
        val rw = (x1 - x0 + 1).coerceAtLeast(1)
        val rh = (y1 - y0 + 1).coerceAtLeast(1)

        // Sample a coarse grid inside an ellipse to reduce background influence.
        val step = (minOf(rw, rh) / 64).coerceIn(2, 10)
        val cx = (x0 + x1) * 0.5f
        val cy = (y0 + y1) * 0.5f
        val rx = (rw * 0.48f).coerceAtLeast(6f)
        val ry = (rh * 0.48f).coerceAtLeast(6f)

        var sum = 0.0
        var sum2 = 0.0
        var n = 0
        var clip = 0
        var sumL = 0.0
        var nL = 0
        var sumR = 0.0
        var nR = 0

        fun sampleY(px: Int, py: Int): Float {
            val base = (py * width + px) * 4
            val r = (rgba.get(base).toInt() and 0xFF) / 255f
            val g = (rgba.get(base + 1).toInt() and 0xFF) / 255f
            val b = (rgba.get(base + 2).toInt() and 0xFF) / 255f
            return (r * 0.2126f + g * 0.7152f + b * 0.0722f).coerceIn(0f, 1f)
        }

        var y = y0
        while (y <= y1) {
            var x = x0
            while (x <= x1) {
                val dx = (x.toFloat() - cx) / rx
                val dy = (y.toFloat() - cy) / ry
                if (dx * dx + dy * dy <= 1.0f) {
                    val Y = sampleY(x, y)
                    // Reject extreme outliers (rare, but helps in very noisy/dim frames).
                    if (Y in 0.02f..0.98f) {
                        sum += Y.toDouble()
                        sum2 += (Y * Y).toDouble()
                        n++
                        if (Y > 0.95f) clip++
                        if (x.toFloat() < cx) { sumL += Y.toDouble(); nL++ } else { sumR += Y.toDouble(); nR++ }
                    }
                }
                x += step
            }
            y += step
        }

        if (n < 32) return

        val mean = (sum / n.toDouble()).toFloat().coerceIn(0f, 1f)
        val var0 = (sum2 / n.toDouble() - (mean * mean).toDouble()).toFloat().coerceAtLeast(0f)
        val std = kotlin.math.sqrt(var0).coerceIn(0f, 1f)
        val clipFrac = (clip.toFloat() / n.toFloat()).coerceIn(0f, 1f)

        val meanL = if (nL > 0) (sumL / nL.toDouble()).toFloat() else mean
        val meanR0 = if (nR > 0) (sumR / nR.toDouble()).toFloat() else mean
        val diff = (meanL - meanR0).coerceIn(-0.15f, 0.15f)
        val bias = (0.5f + diff * 3.2f).coerceIn(0f, 1f)

        val a = 0.08f
        if (!hasLightingEma) {
            faceMeanYEma = mean
            faceStdYEma = std
            clipFracEma = clipFrac
            lightBiasEma = bias
            hasLightingEma = true
        } else {
            faceMeanYEma += (mean - faceMeanYEma) * a
            faceStdYEma += (std - faceStdYEma) * a
            clipFracEma += (clipFrac - clipFracEma) * a
            lightBiasEma += (bias - lightBiasEma) * a
        }
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

    private fun desaturateColor(c: Color, amount: Float): Color {
        val t = amount.coerceIn(0f, 1f)
        val luma = c.red * 0.2126f + c.green * 0.7152f + c.blue * 0.0722f
        val r = (c.red + (luma - c.red) * t).coerceIn(0f, 1f)
        val g = (c.green + (luma - c.green) * t).coerceIn(0f, 1f)
        val b = (c.blue + (luma - c.blue) * t).coerceIn(0f, 1f)
        return Color(r, g, b, c.alpha)
    }

    private fun lightenColor(c: Color, factor: Float) =
        Color(
            (c.red + (1f - c.red) * factor).coerceIn(0f, 1f),
            (c.green + (1f - c.green) * factor).coerceIn(0f, 1f),
            (c.blue + (1f - c.blue) * factor).coerceIn(0f, 1f),
            c.alpha,
        )

    private fun mixColor(a: Color, b: Color, t0: Float): Color {
        val t = t0.coerceIn(0f, 1f)
        return Color(
            red   = (a.red   + (b.red   - a.red)   * t).coerceIn(0f, 1f),
            green = (a.green + (b.green - a.green) * t).coerceIn(0f, 1f),
            blue  = (a.blue  + (b.blue  - a.blue)  * t).coerceIn(0f, 1f),
            alpha = 1f,
        )
    }

    private fun updateSkinTintEmaFromFrame(rgba: ByteBuffer, width: Int, height: Int, lm: FloatArray) {
        if (width <= 2 || height <= 2) return
        // Require enough landmarks.
        if (lm.size < (LandmarkIndex.RIGHT_CHEEK_CENTER * 3 + 2)) return

        fun sampleAround(
            idx: Int,
            radiusPx: Int,
            stepPx: Int,
            out: FloatArray,
            yMaxNorm: Float = 1f,
            lumaMin: Float = 0.10f,
            lumaMax: Float = 0.90f,
            satMax: Float = 0.35f,
        ): Int {
            val nx = lm[idx * 3].coerceIn(0f, 1f)
            val ny = lm[idx * 3 + 1].coerceIn(0f, 1f)
            val cx = (nx * (width - 1)).toInt()
            val cy = (ny * (height - 1)).toInt()

            var sumR = 0f; var sumG = 0f; var sumB = 0f
            var n = 0

            var dy = -radiusPx
            while (dy <= radiusPx) {
                var dx = -radiusPx
                while (dx <= radiusPx) {
                    val x = (cx + dx).coerceIn(0, width - 1)
                    val y = (cy + dy).coerceIn(0, height - 1)
                    val yNorm = y.toFloat() / (height - 1).toFloat()
                    if (yNorm > yMaxNorm) {
                        dx += stepPx
                        continue
                    }
                    val base = (y * width + x) * 4
                    val r = (rgba.get(base).toInt() and 0xFF) / 255f
                    val g = (rgba.get(base + 1).toInt() and 0xFF) / 255f
                    val b = (rgba.get(base + 2).toInt() and 0xFF) / 255f

                    // Reject very dark/bright pixels and very saturated ones (lips/makeup/edges).
                    val luma = r * 0.2126f + g * 0.7152f + b * 0.0722f
                    val mx = maxOf(r, g, b)
                    val mn = minOf(r, g, b)
                    val sat = mx - mn
                    if (luma in lumaMin..lumaMax && sat <= satMax) {
                        sumR += r; sumG += g; sumB += b
                        n++
                    }

                    dx += stepPx
                }
                dy += stepPx
            }

            if (n <= 0) return 0
            out[0] += sumR; out[1] += sumG; out[2] += sumB
            return n
        }

        val radius = (min(width, height) / 180).coerceIn(2, 7)
        val step = maxOf(1, radius / 2)
        val accumUpper = floatArrayOf(0f, 0f, 0f)
        var countUpper = 0
        val accumCheek = floatArrayOf(0f, 0f, 0f)
        var countCheek = 0

        // Beard-robust skin tint estimate:
        // - Upper-face reference (forehead + bridge) is least affected by beard/stubble.
        // - Cheek samples can be pulled darker/cooler by facial hair; use them only when
        //   they agree with the upper-face reference.
        //
        // yMaxNorm gates out the lower face where beard shadow dominates.
        // Sample a couple of upper-nose / bridge anchors (less likely to include moustache shadow than the tip).
        countUpper += sampleAround(6, radius, step, accumUpper, yMaxNorm = 0.60f, satMax = 0.33f)
        countUpper += sampleAround(197, radius, step, accumUpper, yMaxNorm = 0.60f, satMax = 0.33f)
        for (idx in FOREHEAD_CENTER) {
            countUpper += sampleAround(idx, radius, step, accumUpper, yMaxNorm = 0.56f, satMax = 0.33f)
        }

        // Cheeks (optional contribution). Allow slightly lower y but still avoid lower-face beard zone.
        // We apply a *reference-based luma floor* (computed below) so dark beard/stubble pixels
        // are rejected even if they’re low-saturation and would otherwise look “skin-like”.
        // (This is intentionally aggressive per user request.)
        // Note: lumaMin is filled in after we compute the upper-face reference.

        fun colorFrom(accum: FloatArray, count: Int): Color {
            val inv = 1f / count.toFloat()
            return Color(
                red = (accum[0] * inv).coerceIn(0.02f, 0.98f),
                green = (accum[1] * inv).coerceIn(0.02f, 0.98f),
                blue = (accum[2] * inv).coerceIn(0.02f, 0.98f),
                alpha = 1f,
            )
        }

        if (countUpper < 14) return

        val upper = colorFrom(accumUpper, countUpper)

        fun luma(c: Color) = c.red * 0.2126f + c.green * 0.7152f + c.blue * 0.0722f

        val uL = luma(upper)
        val cheekLumaMin = maxOf(0.10f, (uL - 0.08f).coerceIn(0f, 1f))
        // Now sample cheeks with the aggressive luma floor + tighter y gate.
        countCheek += sampleAround(
            LandmarkIndex.LEFT_CHEEK_CENTER,
            radius,
            step,
            accumCheek,
            yMaxNorm = 0.66f,
            lumaMin = cheekLumaMin,
            satMax = 0.33f,
        )
        countCheek += sampleAround(
            LandmarkIndex.RIGHT_CHEEK_CENTER,
            radius,
            step,
            accumCheek,
            yMaxNorm = 0.66f,
            lumaMin = cheekLumaMin,
            satMax = 0.33f,
        )

        val estimate = if (countCheek >= 14) {
            val cheek = colorFrom(accumCheek, countCheek)
            val cL = luma(cheek)
            // If cheeks still come out darker than upper-face reference, treat it as beard/shadow contamination
            // and snap strongly toward upper-face.
            val delta = (uL - cL).coerceIn(0f, 0.20f)
            val beard = ((delta - 0.008f) / 0.035f).coerceIn(0f, 1f) // more sensitive than before
            if (beard > 0.25f) {
                upper
            } else {
                val upperW = (0.70f + 0.25f * beard).coerceIn(0f, 0.95f)
                mixColor(cheek, upper, upperW)
            }
        } else {
            // If cheeks are too sparse after filtering, rely entirely on upper-face.
            upper
        }

        val prev = skinTintEma
        skinTintEma = if (prev == null) {
            estimate
        } else {
            // Small step (EMA) to keep stable under exposure jitter.
            mixColor(prev, estimate, 0.08f)
        }
    }

    private fun effectiveFoundationTint(style: MakeupStyle): Color {
        val base = skinTintEma ?: style.foundationTint
        // Bias toward the style tint based on how much "base coverage" that style is meant to have.
        val bias = (0.08f + 0.62f * style.foundationCoverage).coerceIn(0f, 0.80f)
        val strength = (style.foundationAlpha * 1.25f).coerceIn(0f, 1f)
        return mixColor(base, style.foundationTint, bias * strength)
    }

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
