package com.ham.app.render

import androidx.compose.ui.graphics.Color
import com.ham.app.data.FACE_OVAL
import com.ham.app.data.LandmarkIndex
import kotlin.math.max
import kotlin.math.min

/**
 * Landmark-to-region mapping helpers for contour.
 *
 * This is intentionally *pure Kotlin* and does not depend on ML Kit classes so the project compiles
 * regardless of whether ML Kit is included. If you use ML Kit, adapt your contour points into
 * [MlKitContoursLike] and call [mapFromMlKitContours].
 */
object ContourLandmarkMapping {

    enum class Side { LEFT, RIGHT }

    /**
     * Simple head pose estimate from MediaPipe 478 landmarks.
     *
     * Coordinate assumptions (MediaPipe FaceLandmarker):
     * - x,y are normalized image coordinates (0..1) with y increasing downward
     * - z is roughly in "image width" units, typically more negative when closer to camera
     *
     * We use this to drive angle-aware contour placement/strength:
     * - pitch: camera below face / looking up → stronger jaw + chin, cheek contour shifts higher
     * - yaw: fade nose contour when turning
     */
    data class HeadPose(
        val pitchRad: Float,
        val yawRad: Float,
        val rollRad: Float,
        val lowAngleT: Float,  // 0..1, 1 = looking up (camera below)
        val highAngleT: Float, // 0..1, 1 = camera above / looking down
        val yawT: Float,       // 0..1, 1 = strong yaw/turn
    )

    fun estimateHeadPoseMediaPipe478(lm: FloatArray): HeadPose {
        fun f(idx: Int, o: Int): Float = lm.getOrNull(idx * 3 + o) ?: 0f
        fun clamp01(x: Float) = x.coerceIn(0f, 1f)

        // Key anchors.
        val lx = f(LandmarkIndex.LEFT_EYE_OUTER, 0)
        val ly = f(LandmarkIndex.LEFT_EYE_OUTER, 1)
        val lz = f(LandmarkIndex.LEFT_EYE_OUTER, 2)
        val rx = f(LandmarkIndex.RIGHT_EYE_OUTER, 0)
        val ry = f(LandmarkIndex.RIGHT_EYE_OUTER, 1)
        val rz = f(LandmarkIndex.RIGHT_EYE_OUTER, 2)

        val fx = f(10, 0) // forehead center (same as FACE_OVAL[0])
        val fy = f(10, 1)
        val fz = f(10, 2)

        val cx = f(LandmarkIndex.CHIN, 0)
        val cy = f(LandmarkIndex.CHIN, 1)
        val cz = f(LandmarkIndex.CHIN, 2)

        // Roll from eye line (2D).
        val roll = kotlin.math.atan2((ry - ly).toDouble(), (rx - lx).toDouble()).toFloat()

        // Yaw from left/right jaw depth asymmetry relative to eye span (3D proxy).
        val jLz = f(234, 2) // left jaw corner near ear
        val jRz = f(454, 2) // right jaw corner near ear
        val eyeDx = (rx - lx).coerceAtLeast(1e-4f)
        val yaw = kotlin.math.atan2(((jRz - jLz).toDouble()), eyeDx.toDouble()).toFloat()

        // Pitch from forehead->chin vector: how much it points toward camera (negative z).
        val dy = (cy - fy).coerceAtLeast(1e-4f)
        val dz = (cz - fz)
        val pitch = kotlin.math.atan2(((-dz).toDouble()), dy.toDouble()).toFloat()

        fun smoothstep(e0: Float, e1: Float, x0: Float): Float {
            val t = ((x0 - e0) / (e1 - e0)).coerceIn(0f, 1f)
            return t * t * (3f - 2f * t)
        }

        val pitchDeg = pitch * (180f / kotlin.math.PI.toFloat())
        val yawDeg = yaw * (180f / kotlin.math.PI.toFloat())

        val lowAngleT = smoothstep(8f, 28f, pitchDeg)
        val highAngleT = smoothstep(8f, 28f, -pitchDeg)
        val yawT = clamp01(kotlin.math.abs(yawDeg) / 35f)

        return HeadPose(
            pitchRad = pitch,
            yawRad = yaw,
            rollRad = roll,
            lowAngleT = lowAngleT,
            highAngleT = highAngleT,
            yawT = yawT,
        )
    }

    data class NoseRegion(
        // Two side lines along the bridge.
        val startLeft: FloatArray, // vec2
        val endLeft: FloatArray,   // vec2
        val startRight: FloatArray,// vec2
        val endRight: FloatArray,  // vec2
        // Tip point (for small under-tip shadow).
        val tip: FloatArray,       // vec2
    )

    /**
     * CHEEK CONTOUR (tapered placement below cheekbone).
     *
     * Output is UV points (0..1, top-left origin), UNMIRRORED.
     *
     * The returned polyline is used as the “ridge” reference; the shader places contour
     * slightly *below* it and stops early (does not reach mouth).
     */
    fun getCheekRegion(lm: FloatArray, side: Side): FloatArray {
        val (templeIdx, cheekboneIdx, medialIdx, mouthIdx) = when (side) {
            Side.LEFT -> arrayOf(127, 116, 101, LandmarkIndex.LIP_LEFT)
            Side.RIGHT -> arrayOf(356, 346, 330, LandmarkIndex.LIP_RIGHT)
        }

        fun uv(idx: Int): FloatArray =
            floatArrayOf(
                lm[idx * 3].coerceIn(0f, 1f),
                lm[idx * 3 + 1].coerceIn(0f, 1f),
            )

        val t = uv(templeIdx)
        val b = uv(cheekboneIdx)
        val m0 = uv(medialIdx)
        val mouth = uv(mouthIdx)

        // Stop halfway toward mouth (critical), so contour never reaches the corner.
        val stopT = 0.55f
        val m = floatArrayOf(
            (m0[0] + (mouth[0] - m0[0]) * 0f).coerceIn(0f, 1f),
            (m0[1] + (mouth[1] - m0[1]) * 0f).coerceIn(0f, 1f),
        )
        // Pull medial anchor slightly back toward cheekbone, then toward temple for a cleaner taper.
        val medial = floatArrayOf(
            (b[0] + (m[0] - b[0]) * stopT).coerceIn(0f, 1f),
            (b[1] + (m[1] - b[1]) * stopT).coerceIn(0f, 1f),
        )

        return floatArrayOf(
            t[0], t[1],
            b[0], b[1],
            medial[0], medial[1],
        )
    }

    /**
     * JAW SLIMMING polyline (ear -> chin -> ear) in UV space, UNMIRRORED.
     */
    fun getJawRegion(lm: FloatArray): FloatArray {
        val jawIdx = intArrayOf(356, 323, 288, 365, 377, 152, 149, 172, 132, 234)
        val out = FloatArray(jawIdx.size * 2)
        for (i in jawIdx.indices) {
            val idx = jawIdx[i]
            out[i * 2] = lm[idx * 3].coerceIn(0f, 1f)
            out[i * 2 + 1] = lm[idx * 3 + 1].coerceIn(0f, 1f)
        }
        return out
    }

    /**
     * NOSE CONTOUR region:
     * - two thin side lines along the bridge
     * - tip point for a small under-tip shadow
     *
     * Output is UV space, UNMIRRORED.
     */
    fun getNoseRegion(lm: FloatArray): NoseRegion {
        fun uv(idx: Int): FloatArray =
            floatArrayOf(
                lm[idx * 3].coerceIn(0f, 1f),
                lm[idx * 3 + 1].coerceIn(0f, 1f),
            )

        // Bridge top derived from inner brow midpoint; end at mid bridge.
        val browInnerL = uv(46)
        val browInnerR = uv(276)
        val top = floatArrayOf(
            ((browInnerL[0] + browInnerR[0]) * 0.5f).coerceIn(0f, 1f),
            ((browInnerL[1] + browInnerR[1]) * 0.5f).coerceIn(0f, 1f),
        )
        val mid = uv(195)

        // Symmetric lateral offset based on face width.
        var minX = 1f
        var maxX = 0f
        for (idx in FACE_OVAL) {
            val x = lm[idx * 3].coerceIn(0f, 1f)
            minX = min(minX, x)
            maxX = max(maxX, x)
        }
        val faceW = (maxX - minX).coerceIn(0.10f, 0.95f)
        val off = (faceW * 0.0105f).coerceIn(0.002f, 0.030f)

        val startLeft = floatArrayOf((top[0] - off).coerceIn(0f, 1f), top[1])
        val endLeft = floatArrayOf((mid[0] - off).coerceIn(0f, 1f), mid[1])
        val startRight = floatArrayOf((top[0] + off).coerceIn(0f, 1f), top[1])
        val endRight = floatArrayOf((mid[0] + off).coerceIn(0f, 1f), mid[1])

        return NoseRegion(
            startLeft = startLeft,
            endLeft = endLeft,
            startRight = startRight,
            endRight = endRight,
            tip = uv(LandmarkIndex.NOSE_TIP),
        )
    }

    data class ContourRegions(
        // UV space (0..1, origin top-left), NOT mirrored.
        val cheekLeft: FloatArray,   // 3 * vec2 (temple -> cheekbone -> medial cheek)
        val cheekRight: FloatArray,  // 3 * vec2
        val jaw: FloatArray,         // N * vec2 polyline (ear -> chin -> ear)
        val forehead: FloatArray,    // N * vec2 polyline (temple -> hairline -> temple)
        val noseBridgeStart: FloatArray, // 2 * vec2 (left,right)
        val noseBridgeEnd: FloatArray,   // 2 * vec2 (left,right)

        // Exclusions
        val eyeLeft: EllipseUv,
        val eyeRight: EllipseUv,
        val browLeft: EllipseUv,
        val browRight: EllipseUv,
        val lips: EllipseUv,
        val noseTip: FloatArray,     // vec2
    )

    data class EllipseUv(
        val center: FloatArray, // vec2
        val axis: FloatArray,   // vec2 unit
        val radii: FloatArray,  // vec2
    )

    /**
     * Map from MediaPipe FaceLandmarker (478*3 float array, x/y in 0..1).
     *
     * Notes:
     * - Output is in UV space and unmirrored; mirroring should be applied by the renderer that
     *   knows whether preview is mirrored.
     */
    fun mapFromMediaPipe478(lm: FloatArray): ContourRegions {
        fun uv(idx: Int): FloatArray =
            floatArrayOf(
                lm[idx * 3].coerceIn(0f, 1f),
                lm[idx * 3 + 1].coerceIn(0f, 1f),
            )

        // Cheek anchors (stable zygomatic ridge points).
        val cheekLeft = floatArrayOf(
            lm[127 * 3], lm[127 * 3 + 1],
            lm[116 * 3], lm[116 * 3 + 1],
            lm[101 * 3], lm[101 * 3 + 1],
        )
        val cheekRight = floatArrayOf(
            lm[356 * 3], lm[356 * 3 + 1],
            lm[346 * 3], lm[346 * 3 + 1],
            lm[330 * 3], lm[330 * 3 + 1],
        )

        // Compact jaw + forehead polylines (match ContourMakeupEffect defaults).
        val jawIdx = intArrayOf(356, 323, 288, 365, 377, 152, 149, 172, 132, 234)
        val jaw = FloatArray(jawIdx.size * 2)
        for (i in jawIdx.indices) {
            val idx = jawIdx[i]
            jaw[i * 2] = lm[idx * 3].coerceIn(0f, 1f)
            jaw[i * 2 + 1] = lm[idx * 3 + 1].coerceIn(0f, 1f)
        }

        val foreheadIdx = intArrayOf(127, 21, 103, 109, 10, 338, 332, 251, 356)
        val forehead = FloatArray(foreheadIdx.size * 2)
        for (i in foreheadIdx.indices) {
            val idx = foreheadIdx[i]
            forehead[i * 2] = lm[idx * 3].coerceIn(0f, 1f)
            forehead[i * 2 + 1] = lm[idx * 3 + 1].coerceIn(0f, 1f)
        }

        // Nose bridge lines: derived from inner brows + mid bridge (matches ContourMakeupEffect).
        val browInnerL = uv(46)
        val browInnerR = uv(276)
        val topCx = ((browInnerL[0] + browInnerR[0]) * 0.5f).coerceIn(0f, 1f)
        val topCy = ((browInnerL[1] + browInnerR[1]) * 0.5f).coerceIn(0f, 1f)
        val mid = uv(195)
        // Use a symmetric offset in UV-x based on face width.
        var minX = 1f
        var maxX = 0f
        for (idx in FACE_OVAL) {
            val x = lm[idx * 3].coerceIn(0f, 1f)
            minX = min(minX, x)
            maxX = max(maxX, x)
        }
        val faceW = (maxX - minX).coerceIn(0.10f, 0.95f)
        val off = (faceW * 0.0105f).coerceIn(0.002f, 0.030f)

        val noseStart = floatArrayOf(topCx - off, topCy, topCx + off, topCy)
        val noseEnd = floatArrayOf(mid[0] - off, mid[1], mid[0] + off, mid[1])

        // Simple ellipses for eyes/brows/lips in UV (used primarily for exclusion).
        fun axis(inner: FloatArray, outer: FloatArray): FloatArray {
            val dx = outer[0] - inner[0]
            val dy = outer[1] - inner[1]
            val len = max(1e-6f, kotlin.math.sqrt(dx * dx + dy * dy))
            return floatArrayOf(dx / len, dy / len)
        }

        val eyeLInner = uv(LandmarkIndex.LEFT_EYE_INNER)
        val eyeLOuter = uv(LandmarkIndex.LEFT_EYE_OUTER)
        val eyeRInner = uv(LandmarkIndex.RIGHT_EYE_INNER)
        val eyeROuter = uv(LandmarkIndex.RIGHT_EYE_OUTER)

        fun ellipse(inner: FloatArray, outer: FloatArray, rxK: Float, ryK: Float): EllipseUv {
            val c = floatArrayOf((inner[0] + outer[0]) * 0.5f, (inner[1] + outer[1]) * 0.5f)
            val ax = axis(inner, outer)
            val len = max(1e-6f, kotlin.math.sqrt((outer[0] - inner[0]) * (outer[0] - inner[0]) + (outer[1] - inner[1]) * (outer[1] - inner[1])))
            val r = floatArrayOf((len * rxK).coerceIn(0.01f, 0.45f), (len * ryK).coerceIn(0.01f, 0.45f))
            return EllipseUv(center = c, axis = ax, radii = r)
        }

        val eyeLeft = ellipse(eyeLInner, eyeLOuter, rxK = 0.72f, ryK = 0.45f)
        val eyeRight = ellipse(eyeRInner, eyeROuter, rxK = 0.72f, ryK = 0.45f)

        val browLeft = ellipse(uv(46), uv(105), rxK = 0.72f, ryK = 0.22f)
        val browRight = ellipse(uv(276), uv(334), rxK = 0.72f, ryK = 0.22f)

        val lipL = uv(LandmarkIndex.LIP_LEFT)
        val lipR = uv(LandmarkIndex.LIP_RIGHT)
        val lips = ellipse(lipL, lipR, rxK = 0.80f, ryK = 0.42f)

        val tip = uv(LandmarkIndex.NOSE_TIP)

        return ContourRegions(
            cheekLeft = cheekLeft,
            cheekRight = cheekRight,
            jaw = jaw,
            forehead = forehead,
            noseBridgeStart = noseStart,
            noseBridgeEnd = noseEnd,
            eyeLeft = eyeLeft,
            eyeRight = eyeRight,
            browLeft = browLeft,
            browRight = browRight,
            lips = lips,
            noseTip = tip,
        )
    }

    /**
     * Minimal, dependency-free ML Kit-like contour input.
     *
     * Provide points in UV space (0..1). The caller is responsible for mapping ML Kit coordinates
     * into UV for the analysis frame used by the renderer.
     */
    interface MlKitContoursLike {
        val faceOval: List<PointUv>?
        val leftCheek: List<PointUv>?
        val rightCheek: List<PointUv>?
        val noseBridge: List<PointUv>?
        val noseBottom: List<PointUv>?
        val leftEye: List<PointUv>?
        val rightEye: List<PointUv>?
        val upperLipTop: List<PointUv>?
        val lowerLipBottom: List<PointUv>?
        val leftEyebrowTop: List<PointUv>?
        val rightEyebrowTop: List<PointUv>?
    }

    data class PointUv(val x: Float, val y: Float)

    /**
     * Map from ML Kit-like contour polylines into the region model.
     *
     * This is a best-effort implementation: ML Kit contour sets vary by API version. You may want
     * to tailor which polyline points are used for your exact ML Kit build.
     */
    fun mapFromMlKitContours(c: MlKitContoursLike): ContourRegions? {
        val oval = c.faceOval ?: return null
        if (oval.size < 8) return null

        // Derive face width from oval bounds.
        var minX = 1f; var maxX = 0f
        for (p in oval) {
            minX = min(minX, p.x.coerceIn(0f, 1f))
            maxX = max(maxX, p.x.coerceIn(0f, 1f))
        }
        val faceW = (maxX - minX).coerceIn(0.10f, 0.95f)

        fun pick3Cheek(points: List<PointUv>?): FloatArray {
            // Use an outer->mid->inner ordering if possible; otherwise downsample evenly.
            val pts = (points ?: emptyList())
            if (pts.size >= 3) {
                val a = pts.first()
                val b = pts[pts.size / 2]
                val d = pts.last()
                return floatArrayOf(a.x, a.y, b.x, b.y, d.x, d.y)
            }
            // Fallback: zeroed.
            return FloatArray(6)
        }

        val cheekLeft = pick3Cheek(c.leftCheek)
        val cheekRight = pick3Cheek(c.rightCheek)

        // Jawline: approximate from lower half of face oval (ear->chin->ear).
        val jaw = run {
            val out = FloatArray(10 * 2)
            val start = (oval.size * 0.25f).toInt().coerceIn(0, oval.size - 1)
            val end = (oval.size * 0.75f).toInt().coerceIn(0, oval.size - 1)
            val seg = (end - start).coerceAtLeast(1)
            for (i in 0 until 10) {
                val t = i.toFloat() / 9f
                val idx = (start + (seg * t)).toInt().coerceIn(0, oval.size - 1)
                val p = oval[idx]
                out[i * 2] = p.x.coerceIn(0f, 1f)
                out[i * 2 + 1] = p.y.coerceIn(0f, 1f)
            }
            out
        }

        // Forehead/hairline: approximate from upper half of face oval.
        val forehead = run {
            val out = FloatArray(9 * 2)
            val start = 0
            val end = (oval.size * 0.50f).toInt().coerceIn(1, oval.size - 1)
            val seg = (end - start).coerceAtLeast(1)
            for (i in 0 until 9) {
                val t = i.toFloat() / 8f
                val idx = (start + (seg * t)).toInt().coerceIn(0, oval.size - 1)
                val p = oval[idx]
                out[i * 2] = p.x.coerceIn(0f, 1f)
                out[i * 2 + 1] = p.y.coerceIn(0f, 1f)
            }
            out
        }

        // Nose bridge: pick two points along bridge for start/end; then offset left/right.
        val bridge = c.noseBridge ?: return null
        if (bridge.size < 2) return null
        val top = bridge.first()
        val mid = bridge[bridge.size / 2]
        val off = (faceW * 0.0105f).coerceIn(0.002f, 0.030f)
        val noseStart = floatArrayOf(top.x - off, top.y, top.x + off, top.y)
        val noseEnd = floatArrayOf(mid.x - off, mid.y, mid.x + off, mid.y)

        // Simple exclusions: eyes/brows/lips from their contour polylines.
        fun ellipseFromPolyline(poly: List<PointUv>?, rxK: Float, ryK: Float): EllipseUv {
            val pts = poly ?: emptyList()
            var mnx = 1f; var mxx = 0f; var mny = 1f; var mxy = 0f
            for (p in pts) {
                val x = p.x.coerceIn(0f, 1f)
                val y = p.y.coerceIn(0f, 1f)
                mnx = min(mnx, x); mxx = max(mxx, x)
                mny = min(mny, y); mxy = max(mxy, y)
            }
            val cx = (mnx + mxx) * 0.5f
            val cy = (mny + mxy) * 0.5f
            val rx = max(0.01f, (mxx - mnx) * 0.5f * rxK)
            val ry = max(0.01f, (mxy - mny) * 0.5f * ryK)
            return EllipseUv(
                center = floatArrayOf(cx, cy),
                axis = floatArrayOf(1f, 0f),
                radii = floatArrayOf(rx.coerceIn(0.01f, 0.45f), ry.coerceIn(0.01f, 0.45f)),
            )
        }

        val eyeLeft = ellipseFromPolyline(c.leftEye, rxK = 1.05f, ryK = 1.25f)
        val eyeRight = ellipseFromPolyline(c.rightEye, rxK = 1.05f, ryK = 1.25f)
        val browLeft = ellipseFromPolyline(c.leftEyebrowTop, rxK = 1.10f, ryK = 1.20f)
        val browRight = ellipseFromPolyline(c.rightEyebrowTop, rxK = 1.10f, ryK = 1.20f)

        // Lips: fuse upper/lower to a bbox.
        val lipPts = (c.upperLipTop ?: emptyList()) + (c.lowerLipBottom ?: emptyList())
        val lips = ellipseFromPolyline(lipPts, rxK = 1.15f, ryK = 1.35f)

        // Nose tip: approximate from nose bottom if provided.
        val tip = run {
            val nb = c.noseBottom
            if (nb != null && nb.isNotEmpty()) {
                val p = nb[nb.size / 2]
                floatArrayOf(p.x.coerceIn(0f, 1f), p.y.coerceIn(0f, 1f))
            } else {
                floatArrayOf(mid.x.coerceIn(0f, 1f), mid.y.coerceIn(0f, 1f))
            }
        }

        return ContourRegions(
            cheekLeft = cheekLeft,
            cheekRight = cheekRight,
            jaw = jaw,
            forehead = forehead,
            noseBridgeStart = noseStart,
            noseBridgeEnd = noseEnd,
            eyeLeft = eyeLeft,
            eyeRight = eyeRight,
            browLeft = browLeft,
            browRight = browRight,
            lips = lips,
            noseTip = tip,
        )
    }
}

