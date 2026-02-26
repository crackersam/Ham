package com.ham.app.render

import android.content.Context
import android.graphics.RectF
import android.opengl.GLES20
import android.util.Log
import androidx.compose.ui.graphics.Color
import com.ham.app.R
import com.ham.app.data.FACE_OVAL
import com.ham.app.data.LandmarkIndex
import com.ham.app.render.MakeupGeometry
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Contour makeup module (GPU, GLES2) that generates a landmark-driven mask texture and
 * composites it as a realistic soft shadow (multiply / soft-light).
 *
 * Design goals (per requirements):
 * - TikTok-perfect: soft, diffused, warm-neutral contour with stable placement during motion.
 * - Landmark clamping: never overlaps eyes/lips (soft exclusion ellipses).
 * - Real-time: mask rendered at a reduced resolution + optional temporal smoothing.
 *
 * Integration:
 * - Call [onSurfaceCreated] / [onSurfaceChanged] from the renderer.
 * - Each frame: call [buildContourMask] then [renderContour].
 */
class ContourMakeupEffect(private val context: Context) {
    companion object {
        private const val TAG = "ContourMakeupEffect"
        private const val JAW_COUNT = 10
        private const val FOREHEAD_COUNT = 9
    }

    private var ready = false
    fun isReady(): Boolean = ready

    // ── GL programs ─────────────────────────────────────────────────────────
    private var maskProg = 0
    private var faceMaskProg = 0
    private var compProg = 0 // multiply/softlight composite pass
    private var mixProg = 0
    private var blurProg = 0

    // Mask shader attribs/uniforms.
    private var aPos = 0
    private var aUv = 0
    private var uCropScaleMask = 0
    private var uEnable = 0
    private var uOpacity = 0
    private var uStrength = 0
    private var uSigmaPx = 0
    private var uMaskSize = 0
    private var uFaceWidthPx = 0
    private var uLowAngleT = -1
    private var uFaceMaskTex = 0
    private var uFaceMaskSize = 0
    private var uErodePx = 0

    private var uCheekPts = 0
    private var uJawPts = 0
    private var uNoseStart = 0
    private var uNoseEnd = 0
    private var uForeheadPts = 0
    private var uFaceCenter = 0
    private var uSideVis = 0

    private var uEyeCenter = 0
    private var uEyeAxis = 0
    private var uEyeRadii = 0
    private var uBrowCenter = 0
    private var uBrowAxis = 0
    private var uBrowRadii = 0
    private var uLipCenter = 0
    private var uLipRadii = 0
    private var uNoseTip = 0
    private var uNostrilCenter = 0
    private var uNostrilRadii = 0

    // Face-mask shader attribs/uniforms.
    private var aFacePos = 0
    private var aFaceEdge = 0
    private var uCropScaleFace = 0

    // Composite shader attribs/uniforms.
    private var aPosC = 0
    private var aUvC = 0
    private var uCompMaskTex = 0
    private var uCompFrameTex = 0
    private var uCompSkinRgb = 0
    private var uCompCoolTone = 0
    private var uCompIntensity = 0
    private var uCompMaster = 0
    private var uCompBlendMode = 0
    private var uCompUvTransform = -1

    // Blur shader attribs/uniforms (mask-only blur).
    private var aPosB = 0
    private var aUvB = 0
    private var uBlurTex = 0
    private var uBlurTexelSize = 0
    private var uBlurDir = 0
    private var uBlurUvTransform = -1

    // Mix shader attribs/uniforms.
    private var aPosM = 0
    private var aUvM = 0
    private var uMixTexNew = 0
    private var uMixTexPrev = 0
    private var uMixT = 0
    private var uMixUvTransform = -1

    // Mask program: shared UV transform (identity unless overridden).
    private var uMaskUvTransform = -1

    // ── Quad VBO (x, y, u, v) ───────────────────────────────────────────────
    private var quadVbo = 0
    private var geomVbo = 0

    // ── Mask FBO + ping-pong textures ───────────────────────────────────────
    private var fbo = 0
    private var maskTexA = 0
    private var maskTexB = 0
    private var maskTexTmp = 0
    private var faceMaskTex = 0
    private var maskW = 0
    private var maskH = 0
    private var prevIsA = true

    // Shared UV transform used in all sampling passes (identity unless overridden).
    private val uvTransform = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f,
    )

    fun setUvTransform3x3(m: FloatArray) {
        if (m.size < 9) return
        System.arraycopy(m, 0, uvTransform, 0, 9)
    }

    // Geometry staging for face mask mesh (same 5-float layout as MakeupGeometry).
    private val geomBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(16_384).apply { order(ByteOrder.nativeOrder()) }
    private val geomFloatBuffer: FloatBuffer = geomBuffer.asFloatBuffer()

    private val faceVerts: FloatArray = FloatArray(FACE_OVAL.size * 3 * 5) // fan mesh: n triangles * 3 verts * stride(5)

    // ── CPU-side smoothing state (EMA on derived anchors) ───────────────────
    private var lastNs = 0L
    private val cheekPtsSm = FloatArray(12) // 6 * vec2
    private val jawPtsSm = FloatArray(JAW_COUNT * 2)
    private val noseStartSm = FloatArray(4) // 2 * vec2
    private val noseEndSm = FloatArray(4)   // 2 * vec2
    private val foreheadPtsSm = FloatArray(FOREHEAD_COUNT * 2)
    private val faceCenterSm = FloatArray(2)
    private var faceWidthNdcSm = 0f
    private val eyeCenterSm = FloatArray(4) // 2 eyes
    private val eyeAxisSm = FloatArray(4)
    private val eyeRadiiSm = FloatArray(4)
    private val browCenterSm = FloatArray(4) // 2 brows
    private val browAxisSm = FloatArray(4)
    private val browRadiiSm = FloatArray(4)
    private val lipCenterSm = FloatArray(2)
    private val lipRadiiSm = FloatArray(2)
    private val noseTipSm = FloatArray(2)
    private val nostrilCenterSm = FloatArray(4) // 2 nostrils
    private val nostrilRadiiSm = FloatArray(4)
    private val mouthCornersSm = FloatArray(4) // left,right (2 * vec2)

    private var hasPrev = false

    // Landmark indices chosen for stable “pro contour” placement.
    private val cheekLeftIdx = intArrayOf(127, 116, 101)   // near upper ear/temple → cheekbone → medial cheek
    private val cheekRightIdx = intArrayOf(356, 346, 330)

    // Jawline chain (downsampled): ear → chin → ear.
    // Keep it compact to stay within conservative GLES2 fragment uniform limits.
    private val jawIdx = intArrayOf(
        356, 323, 288, 365, 377,
        152,
        149, 172, 132, 234,
    )

    // Forehead / hairline outer arc (temple → top → temple), downsampled.
    private val foreheadIdx = intArrayOf(
        127, 21, 103, 109, 10, 338, 332, 251, 356,
    )

    // Nose bridge anchor (mid bridge).
    private val noseMidIdx = 195
    // Inner eyebrow points (used to define nose bridge top).
    private val browInnerLeftIdx = 46
    private val browInnerRightIdx = 276
    // Brow outer tail points (approx; used only for brow exclusion mask).
    private val browOuterLeftIdx = 105
    private val browOuterRightIdx = 334

    // Runtime cropScale from the main renderer (letterbox / center-crop).
    private var cropScaleX = 1f
    private var cropScaleY = 1f

    // Per-frame screen-space side visibility (for 3/4 poses).
    private var sideVisL = 1f
    private var sideVisR = 1f
    private var lowAngleTFrame = 0f

    // Main onscreen viewport size (restored after offscreen mask passes).
    private var viewW = 1
    private var viewH = 1

    fun setCropScale(scaleX: Float, scaleY: Float) {
        cropScaleX = scaleX
        cropScaleY = scaleY
    }

    fun onSurfaceCreated() {
        ready = false
        // Quad (x, y, u, v).
        //
        // IMPORTANT: This module renders to and samples from FBO textures (packed masks + base relight).
        // Use standard GL texcoords (v=0 at bottom) to avoid double-flipping when the camera
        // background has already been vertically corrected in the first pass.
        val quadData = floatArrayOf(
            -1f, -1f, 0f, 0f,
            1f, -1f, 1f, 0f,
            -1f, 1f, 0f, 1f,
            1f, 1f, 1f, 1f,
        )
        val vbos = IntArray(1)
        GLES20.glGenBuffers(1, vbos, 0)
        quadVbo = vbos[0]
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        val buf = java.nio.ByteBuffer.allocateDirect(quadData.size * 4)
            .order(java.nio.ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply { put(quadData); position(0) }
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, quadData.size * 4, buf, GLES20.GL_STATIC_DRAW)

        try {
            // Programs.
            maskProg = linkProgram(
                loadRawString(context, R.raw.contour_gradient_mask_vertex),
                loadRawString(context, R.raw.contour_gradient_mask_fragment),
            )
            faceMaskProg = linkProgram(
                loadRawString(context, R.raw.contour_face_mask_vertex),
                loadRawString(context, R.raw.contour_face_mask_fragment),
            )
            compProg = linkProgram(
                loadRawString(context, R.raw.contour_blend_vertex),
                loadRawString(context, R.raw.contour_blend_fragment),
            )
            mixProg = linkProgram(
                loadRawString(context, R.raw.contour_mix_vertex),
                loadRawString(context, R.raw.contour_mix_fragment),
            )
            blurProg = linkProgram(
                loadRawString(context, R.raw.contour_mix_vertex),
                loadRawString(context, R.raw.contour_mask_blur_fragment),
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Contour shader init failed; disabling contour effect", t)
            // Leave ready=false so the renderer can fall back.
            return
        }

        // Mask locations.
        aPos = GLES20.glGetAttribLocation(maskProg, "aPosition")
        aUv = GLES20.glGetAttribLocation(maskProg, "aTexCoord")
        uCropScaleMask = GLES20.glGetUniformLocation(maskProg, "uCropScale")
        uEnable = GLES20.glGetUniformLocation(maskProg, "uEnable")
        uStrength = GLES20.glGetUniformLocation(maskProg, "uStrength")
        uSigmaPx = GLES20.glGetUniformLocation(maskProg, "uSigmaPx")
        uMaskSize = GLES20.glGetUniformLocation(maskProg, "uMaskSize")
        uFaceWidthPx = GLES20.glGetUniformLocation(maskProg, "uFaceWidthPx")
        uLowAngleT = GLES20.glGetUniformLocation(maskProg, "uLowAngleT")
        uFaceMaskTex = GLES20.glGetUniformLocation(maskProg, "uFaceMaskTex")
        uFaceMaskSize = GLES20.glGetUniformLocation(maskProg, "uFaceMaskSize")
        uErodePx = GLES20.glGetUniformLocation(maskProg, "uErodePx")
        uMaskUvTransform = GLES20.glGetUniformLocation(maskProg, "uUvTransform")

        // NOTE: For uniform arrays in GLES2, always query the [0] element.
        uCheekPts = GLES20.glGetUniformLocation(maskProg, "uCheekPath[0]")
        uJawPts = GLES20.glGetUniformLocation(maskProg, "uJawPts[0]")
        uNoseStart = GLES20.glGetUniformLocation(maskProg, "uNoseStart[0]")
        uNoseEnd = GLES20.glGetUniformLocation(maskProg, "uNoseEnd[0]")
        uForeheadPts = GLES20.glGetUniformLocation(maskProg, "uForeheadPts[0]")
        uFaceCenter = GLES20.glGetUniformLocation(maskProg, "uFaceCenter")
        uSideVis = GLES20.glGetUniformLocation(maskProg, "uSideVis")

        uEyeCenter = GLES20.glGetUniformLocation(maskProg, "uEyeCenter[0]")
        uEyeAxis = GLES20.glGetUniformLocation(maskProg, "uEyeAxis[0]")
        uEyeRadii = GLES20.glGetUniformLocation(maskProg, "uEyeRadii[0]")
        uBrowCenter = GLES20.glGetUniformLocation(maskProg, "uBrowCenter[0]")
        uBrowAxis = GLES20.glGetUniformLocation(maskProg, "uBrowAxis[0]")
        uBrowRadii = GLES20.glGetUniformLocation(maskProg, "uBrowRadii[0]")
        uLipCenter = GLES20.glGetUniformLocation(maskProg, "uLipCenter")
        uLipRadii = GLES20.glGetUniformLocation(maskProg, "uLipRadii")
        uNoseTip = GLES20.glGetUniformLocation(maskProg, "uNoseTip")
        uNostrilCenter = GLES20.glGetUniformLocation(maskProg, "uNostrilCenter[0]")
        uNostrilRadii = GLES20.glGetUniformLocation(maskProg, "uNostrilRadii[0]")

        // Face-mask locations.
        aFacePos = GLES20.glGetAttribLocation(faceMaskProg, "aPosition")
        aFaceEdge = GLES20.glGetAttribLocation(faceMaskProg, "aEdgeFactor")
        uCropScaleFace = GLES20.glGetUniformLocation(faceMaskProg, "uCropScale")

        // Composite locations.
        aPosC = GLES20.glGetAttribLocation(compProg, "aPosition")
        aUvC = GLES20.glGetAttribLocation(compProg, "aTexCoord")
        uCompMaskTex = GLES20.glGetUniformLocation(compProg, "uMaskTex")
        uCompFrameTex = GLES20.glGetUniformLocation(compProg, "uFrameTex")
        uCompSkinRgb = GLES20.glGetUniformLocation(compProg, "uSkinRgb")
        uCompCoolTone = GLES20.glGetUniformLocation(compProg, "uCoolTone")
        uCompIntensity = GLES20.glGetUniformLocation(compProg, "uIntensity")
        uCompMaster = GLES20.glGetUniformLocation(compProg, "uMaster")
        uCompBlendMode = GLES20.glGetUniformLocation(compProg, "uBlendMode")
        uCompUvTransform = GLES20.glGetUniformLocation(compProg, "uUvTransform")

        // Mix locations.
        aPosM = GLES20.glGetAttribLocation(mixProg, "aPosition")
        aUvM = GLES20.glGetAttribLocation(mixProg, "aTexCoord")
        uMixTexNew = GLES20.glGetUniformLocation(mixProg, "uNewTex")
        uMixTexPrev = GLES20.glGetUniformLocation(mixProg, "uPrevTex")
        uMixT = GLES20.glGetUniformLocation(mixProg, "uT")
        uMixUvTransform = GLES20.glGetUniformLocation(mixProg, "uUvTransform")

        // Blur locations.
        aPosB = GLES20.glGetAttribLocation(blurProg, "aPosition")
        aUvB = GLES20.glGetAttribLocation(blurProg, "aTexCoord")
        uBlurTex = GLES20.glGetUniformLocation(blurProg, "uTex")
        uBlurTexelSize = GLES20.glGetUniformLocation(blurProg, "uTexelSize")
        uBlurDir = GLES20.glGetUniformLocation(blurProg, "uDir")
        uBlurUvTransform = GLES20.glGetUniformLocation(blurProg, "uUvTransform")

        // FBO.
        val fbos = IntArray(1)
        GLES20.glGenFramebuffers(1, fbos, 0)
        fbo = fbos[0]

        // Dynamic geometry VBO (for face mask mesh).
        val geomVbos = IntArray(1)
        GLES20.glGenBuffers(1, geomVbos, 0)
        geomVbo = geomVbos[0]

        ready = true
    }

    fun onSurfaceChanged(viewWidth: Int, viewHeight: Int) {
        viewW = max(1, viewWidth)
        viewH = max(1, viewHeight)
        // Mask resolution:
        // - Slightly higher than half-res to reduce banding/quantization in smooth gradients
        //   (especially noticeable in the nose/center-face region under motion).
        // - Cap long edge to keep it real-time on mid devices.
        val baseScale = 0.67f
        var w = max(1, (viewWidth.toFloat() * baseScale).toInt())
        var h = max(1, (viewHeight.toFloat() * baseScale).toInt())
        val maxDim0 = max(w, h).toFloat()
        val capLongEdge = 720f
        if (maxDim0 > capLongEdge) {
            val s = capLongEdge / maxDim0
            w = max(1, (w.toFloat() * s).toInt())
            h = max(1, (h.toFloat() * s).toInt())
        }
        // Ensure a reasonable minimum so the blur doesn’t collapse into visible steps.
        w = max(384, w)
        h = max(384, h)
        ensureMaskTextures(w, h)
    }

    /**
     * Builds/updates the contour mask texture for this frame.
     *
     * @param landmarks MediaPipe 478*3 array (x,y in 0..1).
     * @param faceRect optional face rect in landmark UV space (0..1). If null, we derive it from FACE_OVAL.
     * @return mask texture id (GL_TEXTURE_2D) containing alpha in all channels.
     */
    fun buildContourMask(
        landmarks: FloatArray,
        faceRect: RectF?,
        params: ContourParams,
        isMirrored: Boolean,
        nowNs: Long,
    ): Int {
        if (maskW <= 0 || maskH <= 0) return 0

        val dtSec =
            if (lastNs != 0L) ((nowNs - lastNs).toFloat() / 1_000_000_000f).coerceIn(1e-4f, 0.10f)
            else (1f / 30f)
        lastNs = nowNs

        // Landmark temporal smoothing:
        // smoothed = lerp(prev, curr, params.landmarkSmoothingT) (requested default: 0.20)
        val a = params.landmarkSmoothingT.coerceIn(0.01f, 1.0f)

        // Face bounds + center (for scaling + hairline insets).
        val derivedRectUv = faceRect ?: computeFaceRectUv(landmarks)
        val faceCenterUv = floatArrayOf(
            (derivedRectUv.left + derivedRectUv.right) * 0.5f,
            (derivedRectUv.top + derivedRectUv.bottom) * 0.5f,
        )
        val faceCenterNdcX = uvToNdcX(faceCenterUv[0], isMirrored)
        val faceCenterNdcY = uvToNdcY(faceCenterUv[1])
        ema2(faceCenterSm, faceCenterNdcX, faceCenterNdcY, a, initIfMissing = !hasPrev)

        // Face width in *mask pixels* (stable scale for offsets/widths).
        val faceWidthNdc =
            (uvToNdcX(derivedRectUv.right, isMirrored) - uvToNdcX(derivedRectUv.left, isMirrored))
                .let { abs(it) }
                .coerceIn(0.15f, 1.95f)
        if (!hasPrev) {
            faceWidthNdcSm = faceWidthNdc
        } else {
            faceWidthNdcSm += (faceWidthNdc - faceWidthNdcSm) * a
        }
        val pxPerNdcX = maskW.toFloat() * 0.5f
        val faceWidthPx = faceWidthNdc * pxPerNdcX

        lowAngleTFrame = ContourLandmarkMapping.estimateHeadPoseMediaPipe478(landmarks).lowAngleT

        // 3/4-angle robustness: attenuate the far side using landmark Z (depth).
        // uSideVis is screen-left/screen-right, so map anatomical -> screen based on mirroring.
        run {
            val idxL = 234 // anatomical left jaw corner near ear
            val idxR = 454 // anatomical right jaw corner near ear
            val zL = if (idxL * 3 + 2 < landmarks.size) landmarks[idxL * 3 + 2] else 0f
            val zR = if (idxR * 3 + 2 < landmarks.size) landmarks[idxR * 3 + 2] else 0f
            // MediaPipe: more negative z is typically closer to camera; diff magnitude is a yaw proxy.
            val diff = (zR - zL).coerceIn(-0.20f, 0.20f)
            val t = (abs(diff) / 0.060f).coerceIn(0f, 1f)
            val far = (1.0f - 0.42f * t).coerceIn(0.55f, 1.0f)
            val near = 1.0f
            val anatLeftNear = (diff > 0f) // right further => left is nearer
            val anatL = if (anatLeftNear) near else far
            val anatR = if (anatLeftNear) far else near
            if (isMirrored) {
                sideVisL = anatR
                sideVisR = anatL
            } else {
                sideVisL = anatL
                sideVisR = anatR
            }
        }

        // Face basis (for consistent “under-cheekbone” offset).
        val lEx = uvToNdcX(landmarks[LandmarkIndex.LEFT_EYE_OUTER * 3].coerceIn(0f, 1f), isMirrored)
        val lEy = uvToNdcY(landmarks[LandmarkIndex.LEFT_EYE_OUTER * 3 + 1].coerceIn(0f, 1f))
        val rEx = uvToNdcX(landmarks[LandmarkIndex.RIGHT_EYE_OUTER * 3].coerceIn(0f, 1f), isMirrored)
        val rEy = uvToNdcY(landmarks[LandmarkIndex.RIGHT_EYE_OUTER * 3 + 1].coerceIn(0f, 1f))
        var axisX = (rEx - lEx)
        var axisY = (rEy - lEy)
        val axisLen = sqrt(axisX * axisX + axisY * axisY).coerceAtLeast(1e-6f)
        axisX /= axisLen; axisY /= axisLen
        // Perp; choose NDC-down direction (negative Y).
        var downX = axisY
        var downY = -axisX
        if (downY > 0f) { downX = -downX; downY = -downY }

        // Cheek contour paths (3 points per side), pushed slightly downward in face space.
        fillCheekPts(
            lm = landmarks,
            mirror = isMirrored,
            a = a,
            initIfMissing = !hasPrev,
        )

        // Jawline polyline.
        fillPolyline(jawPtsSm, landmarks, isMirrored, jawIdx, a, initIfMissing = !hasPrev)

        // Forehead/hairline polyline (outer arc).
        fillPolyline(foreheadPtsSm, landmarks, isMirrored, foreheadIdx, a, initIfMissing = !hasPrev)

        // Nose side lines (two segments).
        fillNoseLines(
            lm = landmarks,
            mirror = isMirrored,
            faceWidthPx = faceWidthPx,
            a = a,
            initIfMissing = !hasPrev,
        )

        // Exclusions (eyes + lips).
        fillEyeEllipses(landmarks, isMirrored, a, initIfMissing = !hasPrev)
        fillBrowEllipses(landmarks, isMirrored, a, initIfMissing = !hasPrev)
        fillLipEllipse(landmarks, isMirrored, a, initIfMissing = !hasPrev)
        run {
            val lx = uvToNdcX(landmarks[LandmarkIndex.LIP_LEFT * 3].coerceIn(0f, 1f), isMirrored)
            val ly = uvToNdcY(landmarks[LandmarkIndex.LIP_LEFT * 3 + 1].coerceIn(0f, 1f))
            val rx = uvToNdcX(landmarks[LandmarkIndex.LIP_RIGHT * 3].coerceIn(0f, 1f), isMirrored)
            val ry = uvToNdcY(landmarks[LandmarkIndex.LIP_RIGHT * 3 + 1].coerceIn(0f, 1f))
            ema2At(mouthCornersSm, 0, lx, ly, a, initIfMissing = !hasPrev)
            ema2At(mouthCornersSm, 2, rx, ry, a, initIfMissing = !hasPrev)
        }
        run {
            val tx = uvToNdcX(landmarks[LandmarkIndex.NOSE_TIP * 3].coerceIn(0f, 1f), isMirrored)
            val ty = uvToNdcY(landmarks[LandmarkIndex.NOSE_TIP * 3 + 1].coerceIn(0f, 1f))
            ema2(noseTipSm, tx, ty, a, initIfMissing = !hasPrev)
        }
        fillNostrilEllipses(
            a = a,
            initIfMissing = !hasPrev,
            faceWidthPx = faceWidthPx,
        )

        // Face size scaling -> blur radius in pixels.
        // Heavy feather is the key to premium contour (TikTok/Snap look).
        val blurRadiusPx = (faceWidthPx * 0.040f).coerceIn(
            faceWidthPx * 0.025f,
            min(faceWidthPx * 0.060f, min(maskW, maskH) * 0.18f),
        )
        val softnessK = params.softness.coerceIn(0f, 1f)
        // Softness is a gentle multiplier only (avoid "double blur" look).
        val sigmaPx = blurRadiusPx * (0.88f + 0.42f * softnessK)

        // Face clip mask erosion in pixels: faceWidthPx * 0.01f (tunable).
        val erosionPx = (faceWidthPx * params.faceErosionScale.coerceIn(0.0f, 0.05f))
            .coerceIn(0.0f, min(maskW, maskH) * 0.08f)

        // Pass A: render strict face clip mask to texture (used to stop background bleed).
        renderFaceMaskToTexture(
            landmarks = landmarks,
            isMirrored = isMirrored,
        )

        val prevTex = if (prevIsA) maskTexA else maskTexB
        val outTex = if (prevIsA) maskTexB else maskTexA

        // Render raw packed mask into temp.
        renderMaskToTexture(
            outTex = maskTexTmp,
            params = params,
            sigmaPx = sigmaPx,
            faceWidthPx = faceWidthPx,
            erosionPx = erosionPx,
        )

        // Mask-only blur (separable). Run 2 iterations to kill blotches/banding.
        // temp -> outTex (H), outTex -> temp (V), then repeat.
        repeat(2) {
            blurMask(outTex = outTex, inTex = maskTexTmp, dirX = 1f, dirY = 0f)
            blurMask(outTex = maskTexTmp, inTex = outTex, dirX = 0f, dirY = 1f)
        }

        // Temporal stabilization (EMA) on the *blurred packed mask*.
        // outTex = lerp(prevTex, maskTexTmp, t)
        val t = params.temporalSmoothing.coerceIn(0f, 1f)
        if (hasPrev && t < 0.999f) {
            mixTextures(outTex = outTex, newTex = maskTexTmp, prevTex = prevTex, t = t)
        } else {
            // First frame or smoothing disabled: copy new -> out.
            mixTextures(outTex = outTex, newTex = maskTexTmp, prevTex = maskTexTmp, t = 1f)
        }

        hasPrev = true
        prevIsA = !prevIsA
        return outTex
    }

    /**
     * Composite the contour onto the currently bound framebuffer.
     *
     * Multiply mode:
     * - Uses GL blend equation (DST_COLOR, ZERO) so underlying rendering (foundation + other makeup)
     *   is preserved and the contour reads as a true shadow.
     */
    fun renderContour(
        frameTexture: Int,
        maskTexture: Int,
        params: ContourParams,
        baseContour: Float,
        baseHighlight: Float,
        baseMicro: Float,
        baseSpec: Float,
        faceMeanY: Float,
        faceStdY: Float,
        clipFrac: Float,
        lightBias: Float,
        frameTexelSizeX: Float,
        frameTexelSizeY: Float,
    ) {
        if (maskTexture == 0) return

        GLES20.glUseProgram(compProg)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        GLES20.glEnableVertexAttribArray(aPosC)
        GLES20.glVertexAttribPointer(aPosC, 2, GLES20.GL_FLOAT, false, 4 * 4, 0)
        GLES20.glEnableVertexAttribArray(aUvC)
        GLES20.glVertexAttribPointer(aUvC, 2, GLES20.GL_FLOAT, false, 4 * 4, 2 * 4)

        // Per-region intensities already include adaptive tuning (passed by ContourRenderer).
        GLES20.glUniform4f(
            uCompIntensity,
            params.cheekContour.coerceIn(0f, 1f),
            params.jawContour.coerceIn(0f, 1f),
            params.noseContour.coerceIn(0f, 1f),
            params.chinContour.coerceIn(0f, 1f),
        )
        GLES20.glUniform1f(uCompMaster, params.intensity.coerceIn(0f, 1f))
        GLES20.glUniform1f(uCompCoolTone, params.coolTone.coerceIn(0f, 1f))
        GLES20.glUniform1f(uCompBlendMode, if (params.blendMode == ContourBlendMode.SOFT_LIGHT) 1f else 0f)
        if (uCompUvTransform >= 0) {
            GLES20.glUniformMatrix3fv(uCompUvTransform, 1, false, uvTransform, 0)
        }
        // Skin sample drives shade in shader; shade uniform kept for future extension.
        GLES20.glUniform3f(uCompSkinRgb, params.shade.red, params.shade.green, params.shade.blue)

        // Mask texture.
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, maskTexture)
        GLES20.glUniform1i(uCompMaskTex, 1)

        // Frame texture.
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, frameTexture)
        GLES20.glUniform1i(uCompFrameTex, 0)

        // Full overwrite: we output the composited base frame.
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(aPosC)
        GLES20.glDisableVertexAttribArray(aUvC)
    }

    // ── Internal: mask rendering ─────────────────────────────────────────────

    private fun renderMaskToTexture(outTex: Int, params: ContourParams, sigmaPx: Float, faceWidthPx: Float, erosionPx: Float) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            outTex,
            0,
        )
        GLES20.glViewport(0, 0, maskW, maskH)
        GLES20.glClearColor(0f, 0f, 0f, 0f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        GLES20.glUseProgram(maskProg)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 4 * 4, 0)
        GLES20.glEnableVertexAttribArray(aUv)
        GLES20.glVertexAttribPointer(aUv, 2, GLES20.GL_FLOAT, false, 4 * 4, 2 * 4)

        GLES20.glUniform2f(uCropScaleMask, cropScaleX, cropScaleY)
        if (uMaskUvTransform >= 0) {
            GLES20.glUniformMatrix3fv(uMaskUvTransform, 1, false, uvTransform, 0)
        }
        GLES20.glUniform4f(
            uEnable,
            if (params.cheekContour > 0.001f) 1f else 0f,
            if (params.jawContour > 0.001f) 1f else 0f,
            if (params.noseContour > 0.001f) 1f else 0f,
            if (params.chinContour > 0.001f) 1f else 0f,
        )
        GLES20.glUniform4f(
            uStrength,
            1.0f, // keep mask normalized; strengths applied in composite
            1.0f,
            1.0f,
            1.0f,
        )
        GLES20.glUniform1f(uSigmaPx, sigmaPx.coerceIn(0.5f, 128f))
        GLES20.glUniform2f(uMaskSize, maskW.toFloat(), maskH.toFloat())
        GLES20.glUniform1f(uFaceWidthPx, faceWidthPx.coerceIn(1.0f, 4096f))
        if (uLowAngleT >= 0) {
            GLES20.glUniform1f(uLowAngleT, lowAngleTFrame.coerceIn(0f, 1f))
        }

        // Strict face clip mask + erosion.
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, faceMaskTex)
        GLES20.glUniform1i(uFaceMaskTex, 2)
        GLES20.glUniform2f(uFaceMaskSize, maskW.toFloat(), maskH.toFloat())
        GLES20.glUniform1f(uErodePx, erosionPx.coerceIn(0f, 64f))

        GLES20.glUniform2fv(uCheekPts, 6, cheekPtsSm, 0)
        GLES20.glUniform2fv(uJawPts, JAW_COUNT, jawPtsSm, 0)
        GLES20.glUniform2fv(uNoseStart, 2, noseStartSm, 0)
        GLES20.glUniform2fv(uNoseEnd, 2, noseEndSm, 0)
        GLES20.glUniform2fv(uForeheadPts, FOREHEAD_COUNT, foreheadPtsSm, 0)
        GLES20.glUniform2f(uFaceCenter, faceCenterSm[0], faceCenterSm[1])
        if (uSideVis >= 0) {
            GLES20.glUniform2f(uSideVis, sideVisL.coerceIn(0f, 1f), sideVisR.coerceIn(0f, 1f))
        }

        GLES20.glUniform2fv(uEyeCenter, 2, eyeCenterSm, 0)
        GLES20.glUniform2fv(uEyeAxis, 2, eyeAxisSm, 0)
        GLES20.glUniform2fv(uEyeRadii, 2, eyeRadiiSm, 0)
        if (uBrowCenter >= 0) {
            GLES20.glUniform2fv(uBrowCenter, 2, browCenterSm, 0)
        }
        if (uBrowAxis >= 0) {
            GLES20.glUniform2fv(uBrowAxis, 2, browAxisSm, 0)
        }
        if (uBrowRadii >= 0) {
            GLES20.glUniform2fv(uBrowRadii, 2, browRadiiSm, 0)
        }
        GLES20.glUniform2f(uLipCenter, lipCenterSm[0], lipCenterSm[1])
        GLES20.glUniform2f(uLipRadii, lipRadiiSm[0], lipRadiiSm[1])
        if (uNoseTip >= 0) {
            GLES20.glUniform2f(uNoseTip, noseTipSm[0], noseTipSm[1])
        }
        if (uNostrilCenter >= 0) {
            GLES20.glUniform2fv(uNostrilCenter, 2, nostrilCenterSm, 0)
        }
        if (uNostrilRadii >= 0) {
            GLES20.glUniform2fv(uNostrilRadii, 2, nostrilRadiiSm, 0)
        }

        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aUv)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        // IMPORTANT: restore the main viewport; otherwise the preview renders in a small corner.
        GLES20.glViewport(0, 0, viewW, viewH)
    }

    private fun renderFaceMaskToTexture(landmarks: FloatArray, isMirrored: Boolean) {
        if (faceMaskTex == 0) return

        // Build face-oval fan mesh (same feather semantics as foundation).
        // Note: MakeupGeometry.lmX/lmY produce unscaled NDC; we apply cropScale in the vertex shader.
        fillFaceFanVerts(faceVerts, landmarks, isMirrored)
        uploadGeometry(faceVerts)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            faceMaskTex,
            0,
        )
        GLES20.glViewport(0, 0, maskW, maskH)
        GLES20.glClearColor(0f, 0f, 0f, 0f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        GLES20.glUseProgram(faceMaskProg)
        GLES20.glUniform2f(uCropScaleFace, cropScaleX, cropScaleY)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, geomVbo)
        val stride = 5 * 4
        GLES20.glEnableVertexAttribArray(aFacePos)
        GLES20.glVertexAttribPointer(aFacePos, 2, GLES20.GL_FLOAT, false, stride, 0)
        GLES20.glEnableVertexAttribArray(aFaceEdge)
        GLES20.glVertexAttribPointer(aFaceEdge, 1, GLES20.GL_FLOAT, false, stride, 2 * 4)

        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, faceVerts.size / 5)

        GLES20.glDisableVertexAttribArray(aFacePos)
        GLES20.glDisableVertexAttribArray(aFaceEdge)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, viewW, viewH)
    }

    private fun fillFaceFanVerts(out: FloatArray, lm: FloatArray, mirrored: Boolean) {
        // Replicate MakeupGeometry.buildFanMesh without allocating.
        val n = FACE_OVAL.size
        var cx = 0f
        var cy = 0f
        for (i in 0 until n) {
            val idx = FACE_OVAL[i]
            cx += MakeupGeometry.lmX(lm, idx, mirrored, 1f)
            cy += MakeupGeometry.lmY(lm, idx)
        }
        cx /= n.toFloat()
        cy /= n.toFloat()

        var vi = 0
        for (i in 0 until n) {
            val next = (i + 1) % n
            val idx0 = FACE_OVAL[i]
            val idx1 = FACE_OVAL[next]

            val x0 = MakeupGeometry.lmX(lm, idx0, mirrored, 1f)
            val y0 = MakeupGeometry.lmY(lm, idx0)
            val x1 = MakeupGeometry.lmX(lm, idx1, mirrored, 1f)
            val y1 = MakeupGeometry.lmY(lm, idx1)

            // Center
            out[vi++] = cx; out[vi++] = cy
            out[vi++] = 1f; out[vi++] = 0.5f; out[vi++] = 0.5f
            // Edge i
            out[vi++] = x0; out[vi++] = y0
            out[vi++] = 0f; out[vi++] = i.toFloat() / n.toFloat(); out[vi++] = 0f
            // Edge next
            out[vi++] = x1; out[vi++] = y1
            out[vi++] = 0f; out[vi++] = next.toFloat() / n.toFloat(); out[vi++] = 0f
        }
    }

    private fun uploadGeometry(verts: FloatArray) {
        geomFloatBuffer.clear()
        geomFloatBuffer.put(verts)
        geomBuffer.position(0).limit(verts.size * 4)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, geomVbo)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, verts.size * 4, geomBuffer, GLES20.GL_DYNAMIC_DRAW)
        geomBuffer.limit(geomBuffer.capacity())
    }

    private fun mixTextures(outTex: Int, newTex: Int, prevTex: Int, t: Float) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            outTex,
            0,
        )
        GLES20.glViewport(0, 0, maskW, maskH)

        GLES20.glUseProgram(mixProg)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        GLES20.glEnableVertexAttribArray(aPosM)
        GLES20.glVertexAttribPointer(aPosM, 2, GLES20.GL_FLOAT, false, 4 * 4, 0)
        GLES20.glEnableVertexAttribArray(aUvM)
        GLES20.glVertexAttribPointer(aUvM, 2, GLES20.GL_FLOAT, false, 4 * 4, 2 * 4)

        if (uMixUvTransform >= 0) {
            GLES20.glUniformMatrix3fv(uMixUvTransform, 1, false, uvTransform, 0)
        }

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, newTex)
        GLES20.glUniform1i(uMixTexNew, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, prevTex)
        GLES20.glUniform1i(uMixTexPrev, 1)

        GLES20.glUniform1f(uMixT, t.coerceIn(0f, 1f))

        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(aPosM)
        GLES20.glDisableVertexAttribArray(aUvM)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        // Restore main viewport for subsequent onscreen draws.
        GLES20.glViewport(0, 0, viewW, viewH)
    }

    private fun blurMask(outTex: Int, inTex: Int, dirX: Float, dirY: Float) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            outTex,
            0,
        )
        GLES20.glViewport(0, 0, maskW, maskH)

        GLES20.glUseProgram(blurProg)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        GLES20.glEnableVertexAttribArray(aPosB)
        GLES20.glVertexAttribPointer(aPosB, 2, GLES20.GL_FLOAT, false, 4 * 4, 0)
        GLES20.glEnableVertexAttribArray(aUvB)
        GLES20.glVertexAttribPointer(aUvB, 2, GLES20.GL_FLOAT, false, 4 * 4, 2 * 4)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(uBlurTex, 0)
        GLES20.glUniform2f(uBlurTexelSize, 1f / maskW.toFloat(), 1f / maskH.toFloat())
        GLES20.glUniform2f(uBlurDir, dirX, dirY)
        if (uBlurUvTransform >= 0) {
            GLES20.glUniformMatrix3fv(uBlurUvTransform, 1, false, uvTransform, 0)
        }

        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(aPosB)
        GLES20.glDisableVertexAttribArray(aUvB)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, viewW, viewH)
    }

    private fun ensureMaskTextures(w: Int, h: Int) {
        if (w == maskW && h == maskH && maskTexA != 0 && maskTexB != 0 && maskTexTmp != 0 && faceMaskTex != 0) return
        maskW = w
        maskH = h

        if (maskTexA != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(maskTexA), 0)
            maskTexA = 0
        }
        if (maskTexB != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(maskTexB), 0)
            maskTexB = 0
        }
        if (maskTexTmp != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(maskTexTmp), 0)
            maskTexTmp = 0
        }
        if (faceMaskTex != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(faceMaskTex), 0)
            faceMaskTex = 0
        }

        maskTexA = createMaskTexture(w, h)
        maskTexB = createMaskTexture(w, h)
        maskTexTmp = createMaskTexture(w, h)
        faceMaskTex = createMaskTexture(w, h)
        hasPrev = false
        prevIsA = true
    }

    private fun createMaskTexture(w: Int, h: Int): Int {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        val id = ids[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
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
        return id
    }

    // ── Internal: derived anchors + smoothing ───────────────────────────────

    private fun computeFaceRectUv(lm: FloatArray): RectF {
        var minX = 1f
        var maxX = 0f
        var minY = 1f
        var maxY = 0f
        for (idx in FACE_OVAL) {
            val x = lm[idx * 3].coerceIn(0f, 1f)
            val y = lm[idx * 3 + 1].coerceIn(0f, 1f)
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
        }
        // Expand slightly to be robust under partial landmark jitter.
        val padX = ((maxX - minX) * 0.06f).coerceIn(0.0f, 0.06f)
        val padY = ((maxY - minY) * 0.06f).coerceIn(0.0f, 0.06f)
        return RectF(
            (minX - padX).coerceIn(0f, 1f),
            (minY - padY).coerceIn(0f, 1f),
            (maxX + padX).coerceIn(0f, 1f),
            (maxY + padY).coerceIn(0f, 1f),
        )
    }

    private fun fillCheekPts(
        lm: FloatArray,
        mirror: Boolean,
        a: Float,
        initIfMissing: Boolean,
    ) {
        // Cheek contour curve (PRIMARY): temple/ear → cheekbone → medial cheek (toward mouth).
        // The contour mask shader applies contour ONLY below this curve, with a fixed offset.
        //
        // Constraint: keep the medial endpoint OUT of the nasolabial fold / mouth-corner zone
        // (avoids "dirty" smile-line shading).
        fun set3(side: Int, p0Idx: Int, p1Idx: Int, p2Idx: Int) {
            val x0 = uvToNdcX(lm[p0Idx * 3].coerceIn(0f, 1f), mirror)
            val y0 = uvToNdcY(lm[p0Idx * 3 + 1].coerceIn(0f, 1f))
            val x1 = uvToNdcX(lm[p1Idx * 3].coerceIn(0f, 1f), mirror)
            val y1 = uvToNdcY(lm[p1Idx * 3 + 1].coerceIn(0f, 1f))
            var x2 = uvToNdcX(lm[p2Idx * 3].coerceIn(0f, 1f), mirror)
            var y2 = uvToNdcY(lm[p2Idx * 3 + 1].coerceIn(0f, 1f))

            // Required: anchor ear → toward mouth, but stop halfway (never reach the corner).
            val mouthIdx = if (side == 0) LandmarkIndex.LIP_LEFT else LandmarkIndex.LIP_RIGHT
            val mouthX = uvToNdcX(lm[mouthIdx * 3].coerceIn(0f, 1f), mirror)
            val mouthY = uvToNdcY(lm[mouthIdx * 3 + 1].coerceIn(0f, 1f))
            val halfX = x1 + (mouthX - x1) * 0.55f
            val halfY = y1 + (mouthY - y1) * 0.55f
            // Blend landmark-stable medial cheek anchor toward the computed halfway point.
            val endBlend = 0.65f
            x2 = x2 + (halfX - x2) * endBlend
            y2 = y2 + (halfY - y2) * endBlend

            // Keep medial point above mouth corner (prevents low-cheek "dirt patch").
            val mouthY0 = mouthY
            val segLen = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)).coerceIn(0.02f, 1.0f)
            val mouthMargin = (segLen * 0.28f).coerceIn(0.010f, 0.085f)
            if (y2 < mouthY0 + mouthMargin) y2 = mouthY0 + mouthMargin

            // Keep medial point from collapsing toward the nose (nasolabial fold zone).
            // If it gets too close to the nose tip in X, pull it back toward cheekbone.
            val noseX = uvToNdcX(lm[LandmarkIndex.NOSE_TIP * 3].coerceIn(0f, 1f), mirror)
            val minNoseDx = (segLen * 0.22f).coerceIn(0.010f, 0.080f)
            if (kotlin.math.abs(x2 - noseX) < minNoseDx) {
                val t = 0.65f
                x2 = x1 + (x2 - x1) * t
                y2 = y1 + (y2 - y1) * t
            }

            val idxBase = side * 3 * 2
            ema2At(cheekPtsSm, idxBase + 0, x0, y0, a, initIfMissing)
            ema2At(cheekPtsSm, idxBase + 2, x1, y1, a, initIfMissing)
            ema2At(cheekPtsSm, idxBase + 4, x2, y2, a, initIfMissing)
        }

        // Use stable mesh landmarks for a sculpt-like diagonal: temple/ear -> cheekbone -> medial cheek.
        // Indices match ContourBlushEffect and known stable zygomatic ridge points.
        set3(0, p0Idx = 127, p1Idx = 116, p2Idx = 101)
        set3(1, p0Idx = 356, p1Idx = 346, p2Idx = 330)
    }

    private fun fillPolyline(
        out: FloatArray,
        lm: FloatArray,
        mirror: Boolean,
        indices: IntArray,
        a: Float,
        initIfMissing: Boolean,
    ) {
        val n = min(out.size / 2, indices.size)
        for (i in 0 until n) {
            val idx = indices[i]
            val x = uvToNdcX(lm[idx * 3].coerceIn(0f, 1f), mirror)
            val y = uvToNdcY(lm[idx * 3 + 1].coerceIn(0f, 1f))
            ema2At(out, i * 2, x, y, a, initIfMissing)
        }
    }

    private fun fillNoseLines(
        lm: FloatArray,
        mirror: Boolean,
        faceWidthPx: Float,
        a: Float,
        initIfMissing: Boolean,
    ) {
        // FINAL nose contour anchors:
        // - Two symmetrical bridge lines only (no center line).
        // - Define centerline from inner eyebrow region -> mid-nose.
        val blX = lm[browInnerLeftIdx * 3].coerceIn(0f, 1f)
        val blY = lm[browInnerLeftIdx * 3 + 1].coerceIn(0f, 1f)
        val brX = lm[browInnerRightIdx * 3].coerceIn(0f, 1f)
        val brY = lm[browInnerRightIdx * 3 + 1].coerceIn(0f, 1f)

        val topCxUv = ((blX + brX) * 0.5f).coerceIn(0f, 1f)
        val topCyUv = ((blY + brY) * 0.5f).coerceIn(0f, 1f)

        val midXUv = lm[noseMidIdx * 3].coerceIn(0f, 1f)
        val midYUv = lm[noseMidIdx * 3 + 1].coerceIn(0f, 1f)

        val topCx = uvToNdcX(topCxUv, mirror)
        val topCy = uvToNdcY(topCyUv)
        val midCx = uvToNdcX(midXUv, mirror)
        val midCy = uvToNdcY(midYUv)

        // Direction (centerline), then pixel-correct perpendicular for symmetric offsets.
        var dxN = (midCx - topCx)
        var dyN = (midCy - topCy)
        val dLenN = sqrt(dxN * dxN + dyN * dyN).coerceAtLeast(1e-6f)
        dxN /= dLenN; dyN /= dLenN

        // Convert direction to pixel space for correct perpendicular even under non-square mask.
        val dxPx = dxN * (maskW.toFloat() * 0.5f)
        val dyPx = dyN * (maskH.toFloat() * 0.5f)
        var px = -dyPx
        var py = dxPx
        val pLen = sqrt(px * px + py * py).coerceAtLeast(1e-6f)
        px /= pLen; py /= pLen
        val perpNdcX = px / (maskW.toFloat() * 0.5f)
        val perpNdcY = py / (maskH.toFloat() * 0.5f)

        // Offset between the two lines (in pixels).
        val offsetPx = (faceWidthPx * 0.0105f).coerceIn(0.75f, 18.0f)
        val offX = perpNdcX * offsetPx
        val offY = perpNdcY * offsetPx

        // Left / right lines, symmetric around the centerline.
        ema2At(noseStartSm, 0, topCx - offX, topCy - offY, a, initIfMissing)
        ema2At(noseEndSm, 0, midCx - offX, midCy - offY, a, initIfMissing)
        ema2At(noseStartSm, 2, topCx + offX, topCy + offY, a, initIfMissing)
        ema2At(noseEndSm, 2, midCx + offX, midCy + offY, a, initIfMissing)
    }

    private fun fillEyeEllipses(lm: FloatArray, mirror: Boolean, a: Float, initIfMissing: Boolean) {
        fun setEye(
            side: Int,
            innerIdx: Int,
            outerIdx: Int,
            lowerIdx: Int,
        ) {
            val ix = uvToNdcX(lm[innerIdx * 3].coerceIn(0f, 1f), mirror)
            val iy = uvToNdcY(lm[innerIdx * 3 + 1].coerceIn(0f, 1f))
            val ox = uvToNdcX(lm[outerIdx * 3].coerceIn(0f, 1f), mirror)
            val oy = uvToNdcY(lm[outerIdx * 3 + 1].coerceIn(0f, 1f))
            val cx = (ix + ox) * 0.5f
            val cy = (iy + oy) * 0.5f

            var ax = ox - ix
            var ay = oy - iy
            val len = sqrt(ax * ax + ay * ay).coerceAtLeast(1e-6f)
            ax /= len; ay /= len

            // Eye height from lower-lid point (conservative).
            val ly = uvToNdcY(lm[lowerIdx * 3 + 1].coerceIn(0f, 1f))
            val eyeH = abs(cy - ly).coerceAtLeast(len * 0.12f)

            val rx = (len * 0.72f).coerceIn(0.010f, 0.42f)
            val ry = (eyeH * 1.65f).coerceIn(0.008f, 0.32f)

            ema2At(eyeCenterSm, side * 2, cx, cy, a, initIfMissing)
            ema2At(eyeAxisSm, side * 2, ax, ay, a, initIfMissing)
            ema2At(eyeRadiiSm, side * 2, rx, ry, a, initIfMissing)
        }

        setEye(0, LandmarkIndex.LEFT_EYE_INNER, LandmarkIndex.LEFT_EYE_OUTER, LandmarkIndex.LEFT_EYE_LOWER)
        setEye(1, LandmarkIndex.RIGHT_EYE_INNER, LandmarkIndex.RIGHT_EYE_OUTER, LandmarkIndex.RIGHT_EYE_LOWER)
    }

    private fun fillBrowEllipses(lm: FloatArray, mirror: Boolean, a: Float, initIfMissing: Boolean) {
        fun setBrow(side: Int, innerIdx: Int, outerIdx: Int) {
            val ix = uvToNdcX(lm[innerIdx * 3].coerceIn(0f, 1f), mirror)
            val iy = uvToNdcY(lm[innerIdx * 3 + 1].coerceIn(0f, 1f))
            val ox = uvToNdcX(lm[outerIdx * 3].coerceIn(0f, 1f), mirror)
            val oy = uvToNdcY(lm[outerIdx * 3 + 1].coerceIn(0f, 1f))

            var ax = ox - ix
            var ay = oy - iy
            val len = sqrt(ax * ax + ay * ay).coerceAtLeast(1e-6f)
            ax /= len; ay /= len

            // Center slightly above the brow line so we exclude brow hair + immediate halo.
            var cx = (ix + ox) * 0.5f
            var cy = (iy + oy) * 0.5f
            cy += (len * 0.10f).coerceIn(0.004f, 0.035f) // NDC up (positive y)

            val rx = (len * 0.72f).coerceIn(0.020f, 0.55f)
            val ry = (len * 0.22f).coerceIn(0.012f, 0.22f)

            ema2At(browCenterSm, side * 2, cx, cy, a, initIfMissing)
            ema2At(browAxisSm, side * 2, ax, ay, a, initIfMissing)
            ema2At(browRadiiSm, side * 2, rx, ry, a, initIfMissing)
        }

        setBrow(0, browInnerLeftIdx, browOuterLeftIdx)
        setBrow(1, browInnerRightIdx, browOuterRightIdx)
    }

    private fun fillNostrilEllipses(a: Float, initIfMissing: Boolean, faceWidthPx: Float) {
        // Derive nostril centers from the nose tip + nose perpendicular (from the already-built bridge line spacing).
        // This is intentionally conservative: it should only prevent nose contour from entering nostrils,
        // not carve out visible holes.
        val tipX = noseTipSm[0]
        val tipY = noseTipSm[1]

        // Nose line endpoints are mid-bridge; use their midpoint to define a centerline direction.
        val cAx = (noseStartSm[0] + noseStartSm[2]) * 0.5f
        val cAy = (noseStartSm[1] + noseStartSm[3]) * 0.5f
        val cBx = (noseEndSm[0] + noseEndSm[2]) * 0.5f
        val cBy = (noseEndSm[1] + noseEndSm[3]) * 0.5f

        var dx = (cBx - cAx)
        var dy = (cBy - cAy)
        val dLen = sqrt(dx * dx + dy * dy).coerceAtLeast(1e-6f)
        dx /= dLen; dy /= dLen
        // Ensure dx,dy points downward in face space (NDC down = negative Y).
        if (dy > 0f) { dx = -dx; dy = -dy }

        // Perpendicular (nose width direction).
        var px = -dy
        var py = dx
        val pLen = sqrt(px * px + py * py).coerceAtLeast(1e-6f)
        px /= pLen; py /= pLen

        val pxPerNdcX = maskW.toFloat() * 0.5f
        val offsetPx = (faceWidthPx * 0.0105f).coerceIn(0.75f, 18.0f) // matches nose bridge line separation
        val offNdc = (offsetPx / pxPerNdcX).coerceIn(0.002f, 0.10f)

        // Nostrils live slightly below the tip, and slightly wider than bridge-line separation.
        val downNdc = ((offsetPx * 0.70f) / pxPerNdcX).coerceIn(0.002f, 0.10f)
        val widen = 1.35f

        val nxL = tipX - px * offNdc * widen + dx * downNdc
        val nyL = tipY - py * offNdc * widen + dy * downNdc
        val nxR = tipX + px * offNdc * widen + dx * downNdc
        val nyR = tipY + py * offNdc * widen + dy * downNdc

        val rx = (offNdc * 0.72f).coerceIn(0.006f, 0.060f)
        val ry = (offNdc * 0.55f).coerceIn(0.006f, 0.060f)

        ema2At(nostrilCenterSm, 0, nxL, nyL, a, initIfMissing)
        ema2At(nostrilCenterSm, 2, nxR, nyR, a, initIfMissing)
        ema2At(nostrilRadiiSm, 0, rx, ry, a, initIfMissing)
        ema2At(nostrilRadiiSm, 2, rx, ry, a, initIfMissing)
    }

    private fun fillLipEllipse(lm: FloatArray, mirror: Boolean, a: Float, initIfMissing: Boolean) {
        val lx = uvToNdcX(lm[LandmarkIndex.LIP_LEFT * 3].coerceIn(0f, 1f), mirror)
        val ly = uvToNdcY(lm[LandmarkIndex.LIP_LEFT * 3 + 1].coerceIn(0f, 1f))
        val rx = uvToNdcX(lm[LandmarkIndex.LIP_RIGHT * 3].coerceIn(0f, 1f), mirror)
        val ry = uvToNdcY(lm[LandmarkIndex.LIP_RIGHT * 3 + 1].coerceIn(0f, 1f))
        val tx = uvToNdcX(lm[LandmarkIndex.LIP_TOP_CENTER * 3].coerceIn(0f, 1f), mirror)
        val ty = uvToNdcY(lm[LandmarkIndex.LIP_TOP_CENTER * 3 + 1].coerceIn(0f, 1f))
        val bx = uvToNdcX(lm[LandmarkIndex.LIP_BOTTOM_CENTER * 3].coerceIn(0f, 1f), mirror)
        val by = uvToNdcY(lm[LandmarkIndex.LIP_BOTTOM_CENTER * 3 + 1].coerceIn(0f, 1f))

        val cx = (lx + rx) * 0.5f
        val cy = (ty + by) * 0.5f
        val radX = (abs(rx - lx) * 0.80f).coerceIn(0.020f, 0.70f)
        val radY = (abs(by - ty) * 1.05f).coerceIn(0.016f, 0.55f)

        ema2(lipCenterSm, cx, cy, a, initIfMissing)
        ema2(lipRadiiSm, radX, radY, a, initIfMissing)
    }

    private fun ema2(out: FloatArray, x: Float, y: Float, a: Float, initIfMissing: Boolean) {
        if (initIfMissing) {
            out[0] = x; out[1] = y
        } else {
            out[0] += (x - out[0]) * a
            out[1] += (y - out[1]) * a
        }
    }

    private fun ema2At(out: FloatArray, i: Int, x: Float, y: Float, a: Float, initIfMissing: Boolean) {
        if (initIfMissing) {
            out[i] = x; out[i + 1] = y
        } else {
            out[i] += (x - out[i]) * a
            out[i + 1] += (y - out[i + 1]) * a
        }
    }

    private fun uvToNdcX(xUv: Float, mirrored: Boolean): Float {
        val x = if (mirrored) 1f - xUv else xUv
        // Must match the cropScale applied in contour_*_vertex.glsl (vNdcPos = aPosition * uCropScale).
        return (x * 2f - 1f) * cropScaleX
    }

    private fun uvToNdcY(yUv: Float): Float =
        // Must match the cropScale applied in contour_*_vertex.glsl.
        (1f - yUv * 2f) * cropScaleY

}

