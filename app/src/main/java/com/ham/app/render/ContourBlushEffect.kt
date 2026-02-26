package com.ham.app.render

import com.ham.app.data.LandmarkIndex
import kotlin.math.sqrt

/**
 * Frame-local effect generator for contour + blush.
 *
 * - Computes a small derived anchor set from MediaPipe landmarks (UV space 0..1).
 * - Applies OneEuro smoothing to ONLY those derived anchors (not all 478 points).
 * - Produces GPU-uniform-ready arrays with stable sizing (GLES2-friendly).
 *
 * Coordinate system:
 * - All outputs are in camera UV space: x=0..1 left→right, y=0..1 top→bottom
 * - Mirror is applied here so shader uniforms match foundation_vertex.glsl vCamUV.
 */
class ContourBlushEffect {

    data class ContourParams(
        val intensity: Float,   // 0..1
        val cheekIntensity: Float, // 0..1 (recommended ~0.18–0.28)
        val jawIntensity: Float,   // 0..1 (recommended ~0.08–0.18)
        val scale: Float,       // 0.5..1.5
        val feather: Float,     // 0..1
        // Small vertical placement adjustment in *face space*:
        // - -1 = slightly higher (toward cheekbone/temple)
        // -  0 = default
        // - +1 = slightly lower (toward jaw)
        //
        // Applied only to contour anchors (NOT exclusions/blush) to avoid muddying features.
        val placement: Float,   // -1..1
        val warmth: Float,      // -1..1
        val tintR: Float,       // ~0.6..1.4 (1 = neutral)
        val tintG: Float,
        val tintB: Float,
        val noseIntensity: Float, // 0..1 (side shadows only)
    )

    data class BlushParams(
        val intensity: Float, // 0..1 (already includes style alpha * multiplier)
        val scale: Float,     // 0.5..1.5
        val feather: Float,   // 0..1
        val lift: Float,      // -1..1 (positive = lifted toward cheekbone/temple)
        val warmth: Float,    // -1..1
        val tintRgb: FloatArray, // 3
    )

    data class UniformPacket(
        val faceScale: Float,
        val fade: Float,              // 0..1 smoothed confidence fade
        val contour: ContourParams,
        val blush: BlushParams,

        val cheekCenters: FloatArray, // 6 * vec2
        val cheekAxisL: FloatArray,   // vec2
        val cheekAxisUpL: FloatArray, // vec2
        val cheekAxisR: FloatArray,   // vec2
        val cheekAxisUpR: FloatArray, // vec2
        val jawPts: FloatArray,       // 8 * vec2
        val noseStart: FloatArray,    // 2 * vec2
        val noseEnd: FloatArray,      // 2 * vec2

        val eyeCenter: FloatArray,    // 2 * vec2
        val eyeAxis: FloatArray,      // 2 * vec2
        val eyeRadii: FloatArray,     // 2 * vec2
        val browCenter: FloatArray,   // 2 * vec2
        val browRadii: FloatArray,    // 2 * vec2
        val lipCenter: FloatArray,    // vec2
        val lipRadii: FloatArray,     // vec2

        val noseTip: FloatArray,      // vec2
        val blushCenters: FloatArray, // 6 * vec2 (3 per side)
    )

    // Cheek axes should be derived from high, stable landmarks (so blush stays correct).
    // These sit near the zygomatic ridge and are robust under expressions.
    private val cheekAxisLeftIdx = intArrayOf(123, 117, 101)
    private val cheekAxisRightIdx = intArrayOf(352, 347, 330)

    // Cheek contour anchors (start→mid→end) used to build a diagonal lifted ribbon.
    //
    // Goal: a stable, high cheekbone-aligned segment (temple → cheekbone → medial cheek),
    // so the shader's distance-to-segment shadow stays face-locked and never reads as a
    // low-cheek "dirt patch".
    private val contourLeftCheekIdx = intArrayOf(127, 116, 101)    // left: temple, cheekbone, outer-nose / inner-upper-cheek
    private val contourRightCheekIdx = intArrayOf(356, 346, 330)   // right: temple, cheekbone, outer-nose / inner-upper-cheek
    // Jaw polyline anchors spanning the jaw corners through the chin.
    // Order is not important for our shader (it shades distance-to-polyline), but we keep the ends at the jaw corners
    // so the shader can emphasize the "back jaw" segments for definition.
    private val contourJawIdx = intArrayOf(356, 454, 361, 397, 377, 152, 172, 234)

    // Jaw-side anchors near the ear (used to position cheek contour BELOW the cheekbone ridge).
    // These are stable silhouette points and provide the “jaw side” anchor requested for placement.
    private val jawSideLeftIdx = 234
    private val jawSideRightIdx = 454

    // Blush anchors: 3 points per side on the zygomatic ridge (inner→outer-ish).
    private val blushLeftIdx = intArrayOf(123, 117, 101)
    private val blushRightIdx = intArrayOf(352, 347, 330)

    private fun meanUvX(lm: FloatArray, indices: IntArray, mirror: Boolean): Float {
        var sx = 0f
        for (idx in indices) sx += lmUvX(lm, idx, mirror)
        return sx / indices.size.toFloat().coerceAtLeast(1f)
    }

    private fun meanUvY(lm: FloatArray, indices: IntArray): Float {
        var sy = 0f
        for (idx in indices) sy += lmUvY(lm, idx)
        return sy / indices.size.toFloat().coerceAtLeast(1f)
    }

    // ── OneEuro smoothing ────────────────────────────────────────────────────
    private class OneEuroFilter(
        private val minCutoff: Float,
        private val beta: Float,
        private val dCutoff: Float,
    ) {
        private var first = true
        private var xPrev = 0f
        private var dxPrev = 0f

        fun reset() {
            first = true
            xPrev = 0f
            dxPrev = 0f
        }

        private fun alpha(cutoffHz: Float, dtSec: Float): Float {
            val c = cutoffHz.coerceAtLeast(1e-3f)
            val dt = dtSec.coerceAtLeast(1e-4f)
            val tau = (1f / (2f * Math.PI.toFloat() * c))
            return (1f / (1f + tau / dt)).coerceIn(0f, 1f)
        }

        fun filter(x: Float, dtSec: Float): Float {
            if (first) {
                first = false
                xPrev = x
                dxPrev = 0f
                return x
            }
            val dt = dtSec.coerceAtLeast(1e-4f)
            val dx = (x - xPrev) / dt
            val aD = alpha(dCutoff, dt)
            val dxHat = dxPrev + (dx - dxPrev) * aD
            val cutoff = minCutoff + beta * kotlin.math.abs(dxHat)
            val a = alpha(cutoff, dt)
            val xHat = xPrev + (x - xPrev) * a
            xPrev = xHat
            dxPrev = dxHat
            return xHat
        }
    }

    private class OneEuroFilter2D(
        minCutoff: Float,
        beta: Float,
        dCutoff: Float,
    ) {
        private val fx = OneEuroFilter(minCutoff, beta, dCutoff)
        private val fy = OneEuroFilter(minCutoff, beta, dCutoff)

        fun reset() {
            fx.reset()
            fy.reset()
        }

        fun filterX(x: Float, dtSec: Float): Float = fx.filter(x, dtSec)
        fun filterY(y: Float, dtSec: Float): Float = fy.filter(y, dtSec)
    }

    // Fixed uniform layout sizes (reused buffers).
    private val cheekCentersSm = FloatArray(6 * 2)
    private val jawPtsSm = FloatArray(8 * 2)
    private val noseStartSm = FloatArray(2 * 2)
    private val noseEndSm = FloatArray(2 * 2)
    private val eyeCenterSm = FloatArray(2 * 2)
    private val eyeAxisSm = FloatArray(2 * 2)
    private val eyeRadiiSm = FloatArray(2 * 2)
    private val browCenterSm = FloatArray(2 * 2)
    private val browRadiiSm = FloatArray(2 * 2)
    private val lipCenterSm = FloatArray(2)
    private val lipRadiiSm = FloatArray(2)
    private val cheekAxisLSm = FloatArray(2)
    private val cheekAxisUpLSm = FloatArray(2)
    private val cheekAxisRSm = FloatArray(2)
    private val cheekAxisUpRSm = FloatArray(2)
    private val noseTipSm = FloatArray(2)
    private val blushCentersSm = FloatArray(6 * 2)

    // OneEuro filters for the subset of uniforms that move each frame.
    private val fCheekCenters = Array(6) { OneEuroFilter2D(minCutoff = 2.2f, beta = 0.08f, dCutoff = 6.0f) }
    private val fJawPts = Array(8) { OneEuroFilter2D(minCutoff = 2.0f, beta = 0.07f, dCutoff = 6.0f) }
    private val fNoseStart = Array(2) { OneEuroFilter2D(minCutoff = 2.0f, beta = 0.10f, dCutoff = 7.0f) }
    private val fNoseEnd = Array(2) { OneEuroFilter2D(minCutoff = 2.0f, beta = 0.10f, dCutoff = 7.0f) }
    // Face scale drives contour widths in UV space; smoothing it reduces "breathing" under head motion.
    private val fFaceScale = OneEuroFilter(minCutoff = 1.6f, beta = 0.12f, dCutoff = 2.5f)

    private val fEyeCenter = Array(2) { OneEuroFilter2D(minCutoff = 2.2f, beta = 0.06f, dCutoff = 6.0f) }
    private val fEyeAxis = Array(2) { OneEuroFilter2D(minCutoff = 2.4f, beta = 0.03f, dCutoff = 6.0f) }
    private val fEyeRadii = Array(2) { OneEuroFilter2D(minCutoff = 2.0f, beta = 0.02f, dCutoff = 6.0f) }

    private val fBrowCenter = Array(2) { OneEuroFilter2D(minCutoff = 2.0f, beta = 0.05f, dCutoff = 6.0f) }
    private val fBrowRadii = Array(2) { OneEuroFilter2D(minCutoff = 1.8f, beta = 0.02f, dCutoff = 6.0f) }

    private val fLipCenter = OneEuroFilter2D(minCutoff = 2.0f, beta = 0.06f, dCutoff = 6.0f)
    private val fLipRadii = OneEuroFilter2D(minCutoff = 1.8f, beta = 0.02f, dCutoff = 6.0f)

    private val fCheekAxisL = OneEuroFilter2D(minCutoff = 2.5f, beta = 0.02f, dCutoff = 6.0f)
    private val fCheekAxisUpL = OneEuroFilter2D(minCutoff = 2.5f, beta = 0.02f, dCutoff = 6.0f)
    private val fCheekAxisR = OneEuroFilter2D(minCutoff = 2.5f, beta = 0.02f, dCutoff = 6.0f)
    private val fCheekAxisUpR = OneEuroFilter2D(minCutoff = 2.5f, beta = 0.02f, dCutoff = 6.0f)

    private val fNoseTip = OneEuroFilter2D(minCutoff = 2.0f, beta = 0.10f, dCutoff = 7.0f)
    private val fBlushCenters = Array(6) { OneEuroFilter2D(minCutoff = 2.2f, beta = 0.08f, dCutoff = 6.0f) }

    private var fade = 0f
    private var lastNs = 0L
    private var lastGoodNs = 0L

    private fun resetFilters() {
        for (f in fCheekCenters) f.reset()
        for (f in fJawPts) f.reset()
        for (f in fNoseStart) f.reset()
        for (f in fNoseEnd) f.reset()
        for (f in fEyeCenter) f.reset()
        for (f in fEyeAxis) f.reset()
        for (f in fEyeRadii) f.reset()
        for (f in fBrowCenter) f.reset()
        for (f in fBrowRadii) f.reset()
        fLipCenter.reset()
        fLipRadii.reset()
        fCheekAxisL.reset()
        fCheekAxisUpL.reset()
        fCheekAxisR.reset()
        fCheekAxisUpR.reset()
        fNoseTip.reset()
        for (f in fBlushCenters) f.reset()
        fFaceScale.reset()
    }

    private fun lmUvX(lm: FloatArray, idx: Int, mirror: Boolean): Float {
        val x0 = lm[idx * 3]
        return if (mirror) 1f - x0 else x0
    }

    private fun lmUvY(lm: FloatArray, idx: Int): Float =
        lm[idx * 3 + 1]

    private fun norm2(x: Float, y: Float): Float = sqrt(x * x + y * y).coerceAtLeast(1e-6f)

    fun update(
        lm: FloatArray,
        mirror: Boolean,
        nowNs: Long,
        trackingValid: Boolean,
        contour: ContourParams,
        blush: BlushParams,
    ): UniformPacket {
        val dtSec =
            if (lastNs != 0L) ((nowNs - lastNs).toFloat() / 1_000_000_000f).coerceIn(1e-4f, 0.10f)
            else (1f / 30f)
        lastNs = nowNs

        // Robust face scale (inter-ocular distance in UV space).
        val lEx = lmUvX(lm, LandmarkIndex.LEFT_EYE_OUTER, mirror)
        val lEy = lmUvY(lm, LandmarkIndex.LEFT_EYE_OUTER)
        val rEx = lmUvX(lm, LandmarkIndex.RIGHT_EYE_OUTER, mirror)
        val rEy = lmUvY(lm, LandmarkIndex.RIGHT_EYE_OUTER)
        val faceScaleRaw = norm2(rEx - lEx, rEy - lEy).coerceIn(0.05f, 0.65f)
        val faceScale = fFaceScale.filter(faceScaleRaw, dtSec).coerceIn(0.05f, 0.65f)

        // Confidence heuristic + fade smoothing.
        // Use raw scale for validity gating (no "laggy" enable/disable on first frames).
        val scaleOk = faceScaleRaw > 0.075f && faceScaleRaw < 0.60f
        val targetFade = if (scaleOk && trackingValid) 1f else 0f
        val tau = if (targetFade > fade) 0.08f else 0.14f
        val aFade = (1f - kotlin.math.exp(-(dtSec / tau).toDouble()).toFloat()).coerceIn(0f, 1f)
        fade += (targetFade - fade) * aFade

        if (scaleOk && trackingValid) lastGoodNs = nowNs
        val lostForSec = if (lastGoodNs == 0L) 999f else ((nowNs - lastGoodNs).toFloat() / 1_000_000_000f)
        if (lostForSec > 0.35f) resetFilters()

        // Nose tip (used for nostril exclusion in shader).
        run {
            val nx0 = lmUvX(lm, LandmarkIndex.NOSE_TIP, mirror)
            val ny0 = lmUvY(lm, LandmarkIndex.NOSE_TIP)
            noseTipSm[0] = fNoseTip.filterX(nx0, dtSec)
            noseTipSm[1] = fNoseTip.filterY(ny0, dtSec)
        }

        // Global contour placement shift in camera UV space (applied ONLY to contour anchors).
        // UV convention: y increases downward.
        var dxEye = rEx - lEx
        var dyEye = rEy - lEy
        val dLen = norm2(dxEye, dyEye)
        dxEye /= dLen; dyEye /= dLen
        // Perp of eye axis; pick the +down direction (positive y).
        var faceDownX = -dyEye
        var faceDownY = dxEye
        if (faceDownY < 0f) { faceDownX = -faceDownX; faceDownY = -faceDownY }
        val place = contour.placement.coerceIn(-1f, 1f)
        // Max shift ~3% of face scale (subtle; avoids "laggy" re-anchoring).
        val shift = (place * faceScale * 0.030f * contour.scale).coerceIn(-0.030f, 0.030f)
        val contourShiftX = faceDownX * shift
        val contourShiftY = faceDownY * shift

        // ── Cheek axes (cheekbone → outer-eye → temple). AxisUp points toward forehead (uv.y decreases).
        fun computeCheekAxis(
            cheekIdx: IntArray,
            outerEyeIdx: Int,
            templeIdx: Int,
            axisOut: FloatArray,
            axisUpOut: FloatArray,
            fAxis: OneEuroFilter2D,
            fUp: OneEuroFilter2D,
        ) {
            val cx = meanUvX(lm, cheekIdx, mirror)
            val cy = meanUvY(lm, cheekIdx)
            val ex0 = lmUvX(lm, outerEyeIdx, mirror)
            val ey0 = lmUvY(lm, outerEyeIdx)
            val tx0 = lmUvX(lm, templeIdx, mirror)
            val ty0 = lmUvY(lm, templeIdx)

            var tdx = tx0 - cx
            var tdy = ty0 - cy
            val tLen = norm2(tdx, tdy)
            tdx /= tLen; tdy /= tLen

            var edx = ex0 - cx
            var edy = ey0 - cy
            val eLen = norm2(edx, edy)
            edx /= eLen; edy /= eLen

            // Temple direction dominates (lift), but blend in cheek→outer-eye to stabilize on extreme poses.
            var ax = tdx * 0.72f + edx * 0.28f
            var ay = tdy * 0.72f + edy * 0.28f
            val aLen = norm2(ax, ay)
            ax /= aLen; ay /= aLen

            var upX = -ay
            var upY = ax
            if (upY > 0f) { upX = -upX; upY = -upY } // ensure “up” points to smaller y

            val axSm0 = fAxis.filterX(ax, dtSec)
            val aySm0 = fAxis.filterY(ay, dtSec)
            val upXSm0 = fUp.filterX(upX, dtSec)
            val upYSm0 = fUp.filterY(upY, dtSec)

            var axN = axSm0
            var ayN = aySm0
            val axLen = norm2(axN, ayN)
            axN /= axLen; ayN /= axLen

            var upN0 = upXSm0
            var upN1 = upYSm0
            val upLen = norm2(upN0, upN1)
            upN0 /= upLen; upN1 /= upLen

            axisOut[0] = axN
            axisOut[1] = ayN
            axisUpOut[0] = upN0
            axisUpOut[1] = upN1
        }

        computeCheekAxis(
            cheekIdx = cheekAxisLeftIdx,
            outerEyeIdx = LandmarkIndex.LEFT_EYE_OUTER,
            templeIdx = 127,
            axisOut = cheekAxisLSm,
            axisUpOut = cheekAxisUpLSm,
            fAxis = fCheekAxisL,
            fUp = fCheekAxisUpL,
        )
        computeCheekAxis(
            cheekIdx = cheekAxisRightIdx,
            outerEyeIdx = LandmarkIndex.RIGHT_EYE_OUTER,
            templeIdx = 356,
            axisOut = cheekAxisRSm,
            axisUpOut = cheekAxisUpRSm,
            fAxis = fCheekAxisR,
            fUp = fCheekAxisUpR,
        )

        // ── Cheek contour anchors (3 per side): build an upward-lifting diagonal ribbon
        // from (cheek start) → (cheek end).
        //
        // CRITICAL placement requirement:
        // - Cheek contour should sit BELOW the cheekbone ridge (not centered on the cheek),
        //   anchored using (nose outer / medial cheek), (cheekbone), and (jaw side near ear).
        //
        // We output 3 points per side: start, mid, end (for the shader’s segment-based ribbon).
        fun fillCheekCenters(
            side: Int, // 0 = left, 1 = right
            idx3: IntArray,
            axisX: FloatArray,
            axisUp: FloatArray,
            out: FloatArray,
            filters: Array<OneEuroFilter2D>,
        ) {
            val axX = axisX[0]
            val axY = axisX[1]
            val upX = axisUp[0]
            val upY = axisUp[1]
            val downX = -upX
            val downY = -upY
            // Base anchors (in camera UV).
            val sx0 = lmUvX(lm, idx3[0], mirror)
            val sy0 = lmUvY(lm, idx3[0])
            val mx0 = lmUvX(lm, idx3[1], mirror)
            val my0 = lmUvY(lm, idx3[1])
            val ex0 = lmUvX(lm, idx3[2], mirror)
            val ey0 = lmUvY(lm, idx3[2])

            // Robust local scale for this cheek segment.
            val segLen = norm2(ex0 - sx0, ey0 - sy0)
            val base = (0.70f * segLen + 0.30f * faceScale).coerceAtLeast(1e-4f)

            // Use jaw-side landmark as the "down" reference so contour stays face-locked
            // and lands *below* the cheekbone ridge (TikTok-style sculpt placement).
            val jawIdx = if (side == 0) jawSideLeftIdx else jawSideRightIdx
            val jx0 = lmUvX(lm, jawIdx, mirror)
            val jy0 = lmUvY(lm, jawIdx)

            // How far jaw corner sits "below" the cheekbone in the local up/down basis.
            val jawDown = ((jx0 - mx0) * downX + (jy0 - my0) * downY).coerceAtLeast(0f)

            // Move the contour centerline a fraction toward the jaw (below cheekbone),
            // but keep it high (avoid mid-cheek placement).
            val dropBase =
                (jawDown * 0.20f)
                    .coerceIn(base * 0.006f, base * 0.060f) * contour.scale

            // Slight outward shift (away from nose) reduces medial/nose-side muddiness.
            val outward = (base * 0.040f * contour.scale).coerceIn(0.001f, 0.030f)

            // Keep the outer/temple anchor very stable so temple contour stays near hairline.
            val dropS = dropBase * 0.08f
            val dropM = dropBase * 0.48f
            val dropE = dropBase * 0.40f

            var sx = (sx0 + downX * dropS + axX * outward * 0.12f).coerceIn(0f, 1f)
            var sy = (sy0 + downY * dropS + axY * outward * 0.12f).coerceIn(0f, 1f)
            var mx = (mx0 + downX * dropM + axX * outward * 0.34f).coerceIn(0f, 1f)
            var my = (my0 + downY * dropM + axY * outward * 0.34f).coerceIn(0f, 1f)
            var ex = (ex0 + downX * dropE + axX * outward * 0.60f).coerceIn(0f, 1f)
            var ey = (ey0 + downY * dropE + axY * outward * 0.60f).coerceIn(0f, 1f)

            // Never extend contour below mouth level (prevents “dirt” shading in the lower cheek).
            val mouthY = if (side == 0) lmUvY(lm, LandmarkIndex.LIP_LEFT) else lmUvY(lm, LandmarkIndex.LIP_RIGHT)
            val margin = (faceScale * 0.015f * contour.scale).coerceIn(0.0025f, 0.018f)
            val maxAllowedY = (mouthY - margin).coerceIn(0f, 1f)
            val maxY = maxOf(sy, my, ey)
            if (maxY > maxAllowedY) {
                val dy = maxY - maxAllowedY
                // Move upward along the cheek basis to keep the contour in the upper cheek.
                val shift = if (downY > 1e-4f) (dy / downY) else dy
                sx = (sx - downX * shift).coerceIn(0f, 1f)
                sy = (sy - downY * shift).coerceIn(0f, 1f)
                mx = (mx - downX * shift).coerceIn(0f, 1f)
                my = (my - downY * shift).coerceIn(0f, 1f)
                ex = (ex - downX * shift).coerceIn(0f, 1f)
                ey = (ey - downY * shift).coerceIn(0f, 1f)
            }

            // Global contour placement adjustment (up/down in face space).
            // IMPORTANT: apply AFTER the mouth safety clamp, so placement can't push contour into the lower cheek.
            sx = (sx + contourShiftX).coerceIn(0f, 1f); sy = (sy + contourShiftY).coerceIn(0f, 1f)
            mx = (mx + contourShiftX).coerceIn(0f, 1f); my = (my + contourShiftY).coerceIn(0f, 1f)
            ex = (ex + contourShiftX).coerceIn(0f, 1f); ey = (ey + contourShiftY).coerceIn(0f, 1f)

            val f0 = filters[side * 3 + 0]
            out[(side * 3 + 0) * 2] = f0.filterX(sx, dtSec)
            out[(side * 3 + 0) * 2 + 1] = f0.filterY(sy, dtSec)

            val f1 = filters[side * 3 + 1]
            out[(side * 3 + 1) * 2] = f1.filterX(mx, dtSec)
            out[(side * 3 + 1) * 2 + 1] = f1.filterY(my, dtSec)

            val f2 = filters[side * 3 + 2]
            out[(side * 3 + 2) * 2] = f2.filterX(ex, dtSec)
            out[(side * 3 + 2) * 2 + 1] = f2.filterY(ey, dtSec)
        }
        fillCheekCenters(0, contourLeftCheekIdx, cheekAxisLSm, cheekAxisUpLSm, cheekCentersSm, fCheekCenters)
        fillCheekCenters(1, contourRightCheekIdx, cheekAxisRSm, cheekAxisUpRSm, cheekCentersSm, fCheekCenters)

        // ── Jawline polyline inset slightly toward the nose.
        val noseX = noseTipSm[0]
        val noseY = noseTipSm[1]
        for (i in 0 until 8) {
            val jx0 = lmUvX(lm, contourJawIdx[i], mirror)
            val jy0 = lmUvY(lm, contourJawIdx[i])
            var vx = noseX - jx0
            var vy = noseY - jy0
            val vLen = norm2(vx, vy)
            vx /= vLen; vy /= vLen
            // Slightly stronger inset keeps the elongated jaw gradient inside the face-oval mesh
            // (avoids looking clipped/too close to the silhouette edge).
            val inset = (faceScale * 0.068f * contour.scale).coerceIn(0.004f, 0.070f)
            var jx = (jx0 + vx * inset).coerceIn(0f, 1f)
            var jy = (jy0 + vy * inset).coerceIn(0f, 1f)
            // Apply the same subtle global placement shift to jaw anchors.
            jx = (jx + contourShiftX).coerceIn(0f, 1f)
            jy = (jy + contourShiftY).coerceIn(0f, 1f)
            jawPtsSm[i * 2] = fJawPts[i].filterX(jx, dtSec)
            jawPtsSm[i * 2 + 1] = fJawPts[i].filterY(jy, dtSec)
        }

        // ── Nose side lines (2 segments: left/right) from bridge points with lateral offset.
        val bridgeTopIdx = 168
        val bridgeBottomIdx = 195
        val lInX = lmUvX(lm, LandmarkIndex.LEFT_EYE_INNER, mirror)
        val lInY = lmUvY(lm, LandmarkIndex.LEFT_EYE_INNER)
        val rInX = lmUvX(lm, LandmarkIndex.RIGHT_EYE_INNER, mirror)
        val rInY = lmUvY(lm, LandmarkIndex.RIGHT_EYE_INNER)
        var acrossX = rInX - lInX
        var acrossY = rInY - lInY
        val acrossLen = norm2(acrossX, acrossY)
        acrossX /= acrossLen; acrossY /= acrossLen
        val lateral = (faceScale * 0.090f * contour.scale).coerceIn(0.010f, 0.055f)

        fun setNoseLine(side: Int, sign: Float) {
            val sx0 = lmUvX(lm, bridgeTopIdx, mirror)
            val sy0 = lmUvY(lm, bridgeTopIdx)
            val ex0 = lmUvX(lm, bridgeBottomIdx, mirror)
            val ey0 = lmUvY(lm, bridgeBottomIdx)
            var sx = (sx0 + acrossX * (sign * lateral)).coerceIn(0f, 1f)
            var sy = (sy0 + acrossY * (sign * lateral)).coerceIn(0f, 1f)
            var ex = (ex0 + acrossX * (sign * lateral)).coerceIn(0f, 1f)
            var ey = (ey0 + acrossY * (sign * lateral)).coerceIn(0f, 1f)
            // Apply global placement shift to nose-side contour (keeps it aligned with cheek/jaw adjustments).
            sx = (sx + contourShiftX).coerceIn(0f, 1f); sy = (sy + contourShiftY).coerceIn(0f, 1f)
            ex = (ex + contourShiftX).coerceIn(0f, 1f); ey = (ey + contourShiftY).coerceIn(0f, 1f)
            noseStartSm[side * 2] = fNoseStart[side].filterX(sx, dtSec)
            noseStartSm[side * 2 + 1] = fNoseStart[side].filterY(sy, dtSec)
            noseEndSm[side * 2] = fNoseEnd[side].filterX(ex, dtSec)
            noseEndSm[side * 2 + 1] = fNoseEnd[side].filterY(ey, dtSec)
        }
        setNoseLine(0, -1f)
        setNoseLine(1, 1f)

        // ── Exclusions: eyes, brows, lips (soft masks)
        fun setEye(side: Int, innerIdx: Int, outerIdx: Int, lowerIdx: Int) {
            val ix = lmUvX(lm, innerIdx, mirror)
            val iy = lmUvY(lm, innerIdx)
            val ox = lmUvX(lm, outerIdx, mirror)
            val oy = lmUvY(lm, outerIdx)
            val cx = (ix + ox) * 0.5f
            val cy = (iy + oy) * 0.5f
            var ax = ox - ix
            var ay = oy - iy
            val aLen = norm2(ax, ay)
            ax /= aLen; ay /= aLen
            val ly = lmUvY(lm, lowerIdx)
            val eyeSpan = norm2(ox - ix, oy - iy)
            val eyeH = kotlin.math.abs(ly - cy).coerceAtLeast(eyeSpan * 0.10f)

            val rx = (eyeSpan * 0.66f * contour.scale).coerceIn(0.010f, 0.30f)
            val ry = (eyeH * 1.55f * contour.scale).coerceIn(0.008f, 0.22f)

            val scx = fEyeCenter[side].filterX(cx, dtSec)
            val scy = fEyeCenter[side].filterY(cy, dtSec)
            val sax0 = fEyeAxis[side].filterX(ax, dtSec)
            val say0 = fEyeAxis[side].filterY(ay, dtSec)
            val arx = fEyeRadii[side].filterX(rx, dtSec)
            val ary = fEyeRadii[side].filterY(ry, dtSec)
            var sax = sax0
            var say = say0
            val sLen = norm2(sax, say)
            sax /= sLen; say /= sLen

            eyeCenterSm[side * 2] = scx
            eyeCenterSm[side * 2 + 1] = scy
            eyeAxisSm[side * 2] = sax
            eyeAxisSm[side * 2 + 1] = say
            eyeRadiiSm[side * 2] = arx
            eyeRadiiSm[side * 2 + 1] = ary
        }
        setEye(0, LandmarkIndex.LEFT_EYE_INNER, LandmarkIndex.LEFT_EYE_OUTER, LandmarkIndex.LEFT_EYE_LOWER)
        setEye(1, LandmarkIndex.RIGHT_EYE_INNER, LandmarkIndex.RIGHT_EYE_OUTER, LandmarkIndex.RIGHT_EYE_LOWER)

        // Brows: approximate as ellipses centered on the contour arrays' mean.
        val lBx0 = meanUvX(lm, com.ham.app.data.LEFT_BROW, mirror)
        val lBy0 = meanUvY(lm, com.ham.app.data.LEFT_BROW)
        val rBx0 = meanUvX(lm, com.ham.app.data.RIGHT_BROW, mirror)
        val rBy0 = meanUvY(lm, com.ham.app.data.RIGHT_BROW)

        fun setBrow(side: Int, bx0: Float, by0: Float) {
            val eyeSpan = eyeRadiiSm[side * 2] * 2f / 0.66f
            val rx = (eyeSpan * 0.72f * contour.scale).coerceIn(0.012f, 0.34f)
            val ry = (eyeSpan * 0.30f * contour.scale).coerceIn(0.010f, 0.22f)
            browCenterSm[side * 2] = fBrowCenter[side].filterX(bx0, dtSec)
            browCenterSm[side * 2 + 1] = fBrowCenter[side].filterY(by0, dtSec)
            browRadiiSm[side * 2] = fBrowRadii[side].filterX(rx, dtSec)
            browRadiiSm[side * 2 + 1] = fBrowRadii[side].filterY(ry, dtSec)
        }
        setBrow(0, lBx0, lBy0)
        setBrow(1, rBx0, rBy0)

        // Lips: ellipse from key points.
        run {
            val lx = lmUvX(lm, LandmarkIndex.LIP_LEFT, mirror)
            val ly = lmUvY(lm, LandmarkIndex.LIP_LEFT)
            val rx = lmUvX(lm, LandmarkIndex.LIP_RIGHT, mirror)
            val ry = lmUvY(lm, LandmarkIndex.LIP_RIGHT)
            val tx = lmUvX(lm, LandmarkIndex.LIP_TOP_CENTER, mirror)
            val ty = lmUvY(lm, LandmarkIndex.LIP_TOP_CENTER)
            val bx = lmUvX(lm, LandmarkIndex.LIP_BOTTOM_CENTER, mirror)
            val by = lmUvY(lm, LandmarkIndex.LIP_BOTTOM_CENTER)

            val cx0 = (lx + rx) * 0.5f
            val cy0 = (ty + by) * 0.5f
            val radX = (kotlin.math.abs(rx - lx) * 0.70f * contour.scale).coerceIn(0.015f, 0.45f)
            val radY = (kotlin.math.abs(by - ty) * 0.90f * contour.scale).coerceIn(0.012f, 0.32f)

            lipCenterSm[0] = fLipCenter.filterX(cx0, dtSec)
            lipCenterSm[1] = fLipCenter.filterY(cy0, dtSec)
            lipRadiiSm[0] = fLipRadii.filterX(radX, dtSec)
            lipRadiiSm[1] = fLipRadii.filterY(radY, dtSec)
        }

        // ── Blush anchors: 3 per side, lifted toward temple (subtle, Gaussian-only in shader).
        fun fillBlush(side: Int, idx3: IntArray, axisX: FloatArray, axisUp: FloatArray) {
            val axX = axisX[0]
            val axY = axisX[1]
            val upX = axisUp[0]
            val upY = axisUp[1]
            val sizeK = blush.scale // size affects the perceived placement window slightly
            val liftK = blush.lift.coerceIn(-1f, 1f)

            for (i in 0 until 3) {
                val x0 = lmUvX(lm, idx3[i], mirror)
                val y0 = lmUvY(lm, idx3[i])
                // Stronger + higher placement: upper cheek, slightly toward outer eye/temple.
                val along0 = (faceScale * (0.052f + 0.018f * i) * sizeK).coerceIn(0.006f, 0.16f)
                val up0 = (faceScale * (0.060f + 0.016f * i) * sizeK).coerceIn(0.006f, 0.16f)
                val along = along0 * (1.0f + 0.18f * liftK)
                val up = up0 * (1.0f + 0.95f * liftK)
                val x = (x0 + axX * along + upX * up).coerceIn(0f, 1f)
                val y = (y0 + axY * along + upY * up).coerceIn(0f, 1f)
                val f = fBlushCenters[side * 3 + i]
                blushCentersSm[(side * 3 + i) * 2] = f.filterX(x, dtSec)
                blushCentersSm[(side * 3 + i) * 2 + 1] = f.filterY(y, dtSec)
            }
        }
        fillBlush(0, blushLeftIdx, cheekAxisLSm, cheekAxisUpLSm)
        fillBlush(1, blushRightIdx, cheekAxisRSm, cheekAxisUpRSm)

        return UniformPacket(
            faceScale = faceScale,
            fade = fade.coerceIn(0f, 1f),
            contour = contour,
            blush = blush,
            cheekCenters = cheekCentersSm,
            cheekAxisL = cheekAxisLSm,
            cheekAxisUpL = cheekAxisUpLSm,
            cheekAxisR = cheekAxisRSm,
            cheekAxisUpR = cheekAxisUpRSm,
            jawPts = jawPtsSm,
            noseStart = noseStartSm,
            noseEnd = noseEndSm,
            eyeCenter = eyeCenterSm,
            eyeAxis = eyeAxisSm,
            eyeRadii = eyeRadiiSm,
            browCenter = browCenterSm,
            browRadii = browRadiiSm,
            lipCenter = lipCenterSm,
            lipRadii = lipRadiiSm,
            noseTip = noseTipSm,
            blushCenters = blushCentersSm,
        )
    }
}

