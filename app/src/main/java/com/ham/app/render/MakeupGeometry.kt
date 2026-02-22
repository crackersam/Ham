package com.ham.app.render

import com.ham.app.data.*

/**
 * Converts face landmark flat array (x,y,z per point, normalised 0-1) into
 * GPU-ready vertex data for each makeup region.
 *
 * Coordinate system:
 *   • Landmark space: x ∈ [0,1] left→right, y ∈ [0,1] top→bottom
 *   • NDC (OpenGL): x ∈ [-1,1] left→right, y ∈ [-1,1] bottom→top
 *
 * Layout per vertex: [ndcX, ndcY, edgeFactor, regionU, regionV]  → 5 floats
 */
object MakeupGeometry {

    private const val STRIDE = 5

    // ── Fan triangulation ────────────────────────────────────────────────────

    /**
     * Builds a fan-triangulated mesh from a closed polygon of landmark indices.
     *
     * The centroid is the "fan hub" and is assigned edgeFactor = 1.0 (fully
     * opaque center).  Edge vertices are assigned edgeFactor = 0.0 so the
     * fragment shader fades them out smoothly.
     */
    fun buildFanMesh(
        lm: FloatArray,
        indices: IntArray,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
    ): FloatArray {
        val pts = indicesTo2D(lm, indices, isMirrored, aspectScale)
        if (pts.size < 3) return FloatArray(0)

        // Compute centroid
        var cx = 0f; var cy = 0f
        for (i in pts.indices step 2) { cx += pts[i]; cy += pts[i + 1] }
        cx /= (pts.size / 2); cy /= (pts.size / 2)

        val n = pts.size / 2
        val verts = FloatArray(n * 3 * STRIDE) // n triangles, 3 verts each

        var vi = 0
        for (i in 0 until n) {
            val next = (i + 1) % n
            // Center (edgeFactor = 1, regionUV = 0.5,0.5)
            verts[vi++] = cx; verts[vi++] = cy
            verts[vi++] = 1f; verts[vi++] = 0.5f; verts[vi++] = 0.5f
            // Cur (edgeFactor = 0, regionU = normalized angle)
            verts[vi++] = pts[i * 2]; verts[vi++] = pts[i * 2 + 1]
            verts[vi++] = 0f; verts[vi++] = i.toFloat() / n; verts[vi++] = 0f
            // Next
            verts[vi++] = pts[next * 2]; verts[vi++] = pts[next * 2 + 1]
            verts[vi++] = 0f; verts[vi++] = next.toFloat() / n; verts[vi++] = 0f
        }
        return verts
    }

    /**
     * Builds a thickened stroke mesh for liner/eyebrow paths.
     * Each segment becomes a rectangle with soft inner edge (edgeFactor = 1)
     * and zero on the outside.
     */
    fun buildStrokeMesh(
        lm: FloatArray,
        indices: IntArray,
        widthNdc: Float,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
    ): FloatArray {
        val pts = indicesTo2D(lm, indices, isMirrored, aspectScale)
        if (pts.size < 4) return FloatArray(0)  // need ≥ 2 points

        val n = pts.size / 2
        val result = mutableListOf<Float>()

        for (i in 0 until n - 1) {
            val x0 = pts[i * 2]; val y0 = pts[i * 2 + 1]
            val x1 = pts[(i + 1) * 2]; val y1 = pts[(i + 1) * 2 + 1]

            // Perpendicular direction
            val dx = x1 - x0; val dy = y1 - y0
            val len = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat().coerceAtLeast(1e-6f)
            val nx = -dy / len * widthNdc
            val ny =  dx / len * widthNdc

            // Four corners of the segment quad
            val ax = x0 + nx; val ay = y0 + ny
            val bx = x0 - nx; val by = y0 - ny
            val cx = x1 + nx; val cy = y1 + ny
            val dx2 = x1 - nx; val dy2 = y1 - ny

            // Triangle 1: a, b, c
            result += floatArrayOf(ax, ay, 0f, 0f, 0f).toList()
            result += floatArrayOf(bx, by, 0f, 1f, 0f).toList()
            result += floatArrayOf(cx, cy, 0f, 0f, 1f).toList()
            // Triangle 2: b, dx2, c
            result += floatArrayOf(bx, by, 0f, 1f, 0f).toList()
            result += floatArrayOf(dx2, dy2, 0f, 1f, 1f).toList()
            result += floatArrayOf(cx, cy, 0f, 0f, 1f).toList()
        }

        return result.toFloatArray()
    }

    /**
     * Builds radial gradient mesh for blush (large soft ellipse).
     * Uses a single fan around a center point with edgeFactor = 1 at center,
     * 0 at rim.
     */
    fun buildBlushMesh(
        centerX: Float, centerY: Float,
        radiusX: Float, radiusY: Float,
        segments: Int = 32,
    ): FloatArray {
        val verts = FloatArray(segments * 3 * STRIDE)
        var vi = 0
        val step = (2.0 * Math.PI / segments).toFloat()
        for (i in 0 until segments) {
            val a0 = step * i
            val a1 = step * (i + 1)
            // Center
            verts[vi++] = centerX; verts[vi++] = centerY
            verts[vi++] = 1f; verts[vi++] = 0.5f; verts[vi++] = 0.5f
            // Edge vertex i
            verts[vi++] = centerX + Math.cos(a0.toDouble()).toFloat() * radiusX
            verts[vi++] = centerY + Math.sin(a0.toDouble()).toFloat() * radiusY
            verts[vi++] = 0f; verts[vi++] = 0f; verts[vi++] = 0f
            // Edge vertex i+1
            verts[vi++] = centerX + Math.cos(a1.toDouble()).toFloat() * radiusX
            verts[vi++] = centerY + Math.sin(a1.toDouble()).toFloat() * radiusY
            verts[vi++] = 0f; verts[vi++] = 1f; verts[vi++] = 0f
        }
        return verts
    }

    // ── Upper-eyelid strip mesh ──────────────────────────────────────────────

    /**
     * Builds a crescent-shaped strip mesh that covers only the upper eyelid
     * area (from the lash line upward into the crease), never touching the
     * eyeball below.
     *
     * Algorithm:
     *  1. The [upperArcIndices] define the upper lash line from inner corner
     *     to outer corner (9 points).
     *  2. An "outer" boundary is generated by pushing each arc point away from
     *     the eye's horizontal centre by [expansionFactor] × the apex radius
     *     (distance from centre to the topmost arc point).
     *  3. A two-ring strip is built:
     *       arc (lash line) with edgeFactor = cornerFade (1 in the middle,
     *           fades to 0 at inner/outer corners)
     *       outer boundary with edgeFactor = 0 (fully transparent above crease)
     *  4. regionU encodes inner→outer gradient (0 at inner corner, 1 at outer).
     */
    fun buildUpperEyelidMesh(
        lm: FloatArray,
        upperArcIndices: IntArray,
        isMirrored: Boolean = true,
        expansionFactor: Float = 0.50f,
        aspectScale: Float = 1f,
    ): FloatArray {
        val arcPts = indicesTo2D(lm, upperArcIndices, isMirrored, aspectScale)
        val n = arcPts.size / 2
        if (n < 2) return FloatArray(0)

        // Eye horizontal centre: midpoint between inner and outer corners
        val cx = (arcPts[0] + arcPts[(n - 1) * 2]) * 0.5f
        val cy = (arcPts[1] + arcPts[(n - 1) * 2 + 1]) * 0.5f

        // Apex radius: distance from centre to the furthest arc point
        var apexRadius = 0f
        for (i in 0 until n) {
            val dx = arcPts[i * 2] - cx
            val dy = arcPts[i * 2 + 1] - cy
            val d = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
            if (d > apexRadius) apexRadius = d
        }
        apexRadius = apexRadius.coerceAtLeast(1e-4f)

        // Outer boundary: push each arc point outward from eye centre
        val outerPts = FloatArray(n * 2)
        for (i in 0 until n) {
            val px = arcPts[i * 2]; val py = arcPts[i * 2 + 1]
            val dx = px - cx; val dy = py - cy
            val dist = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat().coerceAtLeast(1e-6f)
            val push = apexRadius * expansionFactor
            outerPts[i * 2]     = px + dx / dist * push
            outerPts[i * 2 + 1] = py + dy / dist * push
        }

        // Corner fade: smooth ramp so edgeFactor=0 at the very first/last vertex
        // (eye corners), peaks at 1 in the middle of the arc
        fun cornerFade(i: Int): Float {
            val t = i.toFloat() / (n - 1).toFloat()
            val fade = Math.sin((t * Math.PI)).toFloat()   // 0→1→0 bell curve
            return fade.coerceIn(0f, 1f)
        }

        val result = mutableListOf<Float>()
        for (i in 0 until n - 1) {
            val next = i + 1
            val ix0 = arcPts[i * 2];    val iy0 = arcPts[i * 2 + 1]
            val ix1 = arcPts[next * 2]; val iy1 = arcPts[next * 2 + 1]
            val ox0 = outerPts[i * 2];    val oy0 = outerPts[i * 2 + 1]
            val ox1 = outerPts[next * 2]; val oy1 = outerPts[next * 2 + 1]

            // regionU: 0 at inner corner (i=0), 1 at outer corner (i=n-1)
            val u0 = i.toFloat() / (n - 1)
            val u1 = next.toFloat() / (n - 1)

            val ef0 = cornerFade(i)
            val ef1 = cornerFade(next)

            // Triangle 1: arc[i], arc[i+1], outer[i]
            result += floatArrayOf(ix0, iy0, ef0, u0, 0f).toList()
            result += floatArrayOf(ix1, iy1, ef1, u1, 0f).toList()
            result += floatArrayOf(ox0, oy0, 0f,  u0, 1f).toList()
            // Triangle 2: outer[i], arc[i+1], outer[i+1]
            result += floatArrayOf(ox0, oy0, 0f,  u0, 1f).toList()
            result += floatArrayOf(ix1, iy1, ef1, u1, 0f).toList()
            result += floatArrayOf(ox1, oy1, 0f,  u1, 1f).toList()
        }
        return result.toFloatArray()
    }

    // ── Lip ring mesh ────────────────────────────────────────────────────────

    /**
     * Builds an annular (donut) mesh between [outerIndices] and [innerIndices]
     * so lip colour is only applied to the actual lip tissue, never inside the
     * mouth opening.
     *
     * Both rings must have the same number of points with matching vertex order.
     * A midpoint ring is generated at the average of each corresponding pair;
     * this ring carries edgeFactor = 1 (fully pigmented centre of lip tissue)
     * while the outer and inner boundaries carry edgeFactor = 0 (soft edges).
     *
     * Strip A: outer (ef=0) ↔ mid (ef=1)
     * Strip B: mid   (ef=1) ↔ inner (ef=0)
     */
    fun buildLipRingMesh(
        lm: FloatArray,
        outerIndices: IntArray,
        innerIndices: IntArray,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
    ): FloatArray {
        val n = minOf(outerIndices.size, innerIndices.size)
        val outer = indicesTo2D(lm, outerIndices.copyOf(n), isMirrored, aspectScale)
        val inner = indicesTo2D(lm, innerIndices.copyOf(n), isMirrored, aspectScale)

        // Midpoint ring
        val mid = FloatArray(n * 2)
        for (i in 0 until n) {
            mid[i * 2]     = (outer[i * 2]     + inner[i * 2])     * 0.5f
            mid[i * 2 + 1] = (outer[i * 2 + 1] + inner[i * 2 + 1]) * 0.5f
        }

        val result = mutableListOf<Float>()

        // Two-strip build: (outer↔mid) then (mid↔inner)
        for (strip in 0..1) {
            val ringA  = if (strip == 0) outer else mid
            val ringB  = if (strip == 0) mid   else inner
            val efA    = if (strip == 0) 0f    else 1f   // outer=transparent, mid=opaque
            val efB    = if (strip == 0) 1f    else 0f   // mid=opaque, inner=transparent

            for (i in 0 until n) {
                val next = (i + 1) % n
                val ax0 = ringA[i * 2];    val ay0 = ringA[i * 2 + 1]
                val ax1 = ringA[next * 2]; val ay1 = ringA[next * 2 + 1]
                val bx0 = ringB[i * 2];    val by0 = ringB[i * 2 + 1]
                val bx1 = ringB[next * 2]; val by1 = ringB[next * 2 + 1]

                val u0 = i.toFloat() / n
                val u1 = next.toFloat() / n

                // Triangle 1: a[i], a[next], b[i]
                result += floatArrayOf(ax0, ay0, efA, u0, 0f).toList()
                result += floatArrayOf(ax1, ay1, efA, u1, 0f).toList()
                result += floatArrayOf(bx0, by0, efB, u0, 0.5f).toList()
                // Triangle 2: b[i], a[next], b[next]
                result += floatArrayOf(bx0, by0, efB, u0, 0.5f).toList()
                result += floatArrayOf(ax1, ay1, efA, u1, 0f).toList()
                result += floatArrayOf(bx1, by1, efB, u1, 0.5f).toList()
            }
        }
        return result.toFloatArray()
    }

    // ── Eyeshadow mesh ───────────────────────────────────────────────────────

    /**
     * Like buildFanMesh but assigns regionU = 0 at inner corner, 1 at outer,
     * so the gradient shader can apply inner→outer colour.
     */
    fun buildEyeshadowMesh(
        lm: FloatArray,
        indices: IntArray,
        innerIdx: Int,
        outerIdx: Int,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
    ): FloatArray {
        val pts = indicesTo2D(lm, indices, isMirrored, aspectScale)
        if (pts.size < 6) return FloatArray(0)

        val innerX = lmX(lm, innerIdx, isMirrored, aspectScale)
        val innerY = lmY(lm, innerIdx)
        val outerX = lmX(lm, outerIdx, isMirrored, aspectScale)
        val outerY = lmY(lm, outerIdx)
        val refDx = outerX - innerX
        val refDy = outerY - innerY
        val refLen = Math.sqrt((refDx * refDx + refDy * refDy).toDouble()).toFloat().coerceAtLeast(1e-6f)

        var cx = 0f; var cy = 0f
        for (i in pts.indices step 2) { cx += pts[i]; cy += pts[i + 1] }
        cx /= (pts.size / 2); cy /= (pts.size / 2)

        val n = pts.size / 2
        val verts = FloatArray(n * 3 * STRIDE)
        var vi = 0

        for (i in 0 until n) {
            val next = (i + 1) % n

            fun regionU(ex: Float, ey: Float): Float {
                val dot = (ex - innerX) * refDx + (ey - innerY) * refDy
                return (dot / (refLen * refLen)).coerceIn(0f, 1f)
            }

            // Center
            verts[vi++] = cx; verts[vi++] = cy
            verts[vi++] = 0.8f // near center but not fully opaque so gradient shows
            verts[vi++] = regionU(cx, cy); verts[vi++] = 0.5f

            // Current edge
            val ex = pts[i * 2]; val ey = pts[i * 2 + 1]
            verts[vi++] = ex; verts[vi++] = ey
            verts[vi++] = 0f
            verts[vi++] = regionU(ex, ey); verts[vi++] = 0f

            // Next edge
            val nx2 = pts[next * 2]; val ny2 = pts[next * 2 + 1]
            verts[vi++] = nx2; verts[vi++] = ny2
            verts[vi++] = 0f
            verts[vi++] = regionU(nx2, ny2); verts[vi++] = 0f
        }
        return verts
    }

    // ── Measurement helpers ──────────────────────────────────────────────────

    /**
     * Returns the NDC bounding-box half-extents (radiusX, radiusY) for a set
     * of landmark indices — used to size blush ellipses to the actual cheek
     * region rather than a fixed fraction of face width.
     */
    fun landmarkBoundingRadius(lm: FloatArray, indices: IntArray, mirrored: Boolean, aspectScale: Float = 1f): Pair<Float, Float> {
        val pts = indicesTo2D(lm, indices, mirrored, aspectScale)
        var minX = Float.MAX_VALUE;  var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE;  var maxY = -Float.MAX_VALUE
        for (i in pts.indices step 2) {
            if (pts[i]     < minX) minX = pts[i];     if (pts[i]     > maxX) maxX = pts[i]
            if (pts[i + 1] < minY) minY = pts[i + 1]; if (pts[i + 1] > maxY) maxY = pts[i + 1]
        }
        return Pair((maxX - minX) * 0.5f, (maxY - minY) * 0.5f)
    }

    /**
     * Returns the NDC vertical extent of an eye: distance from the topmost
     * point on the upper-lid arc down to the lower-lid landmark.
     * Used to scale eyeliner width and eyeshadow expansion to the actual eye.
     */
    fun eyeHeight(lm: FloatArray, upperArcIndices: IntArray, lowerLidIdx: Int, mirrored: Boolean, aspectScale: Float = 1f): Float {
        val arcPts = indicesTo2D(lm, upperArcIndices, mirrored, aspectScale)
        var maxY = -Float.MAX_VALUE
        for (i in 1 until arcPts.size step 2) if (arcPts[i] > maxY) maxY = arcPts[i]
        val lowerY = lmY(lm, lowerLidIdx)
        return (maxY - lowerY).coerceAtLeast(0.001f)
    }

    /**
     * Returns the apex radius of an upper-lid arc: the maximum NDC distance
     * from the arc's horizontal centre to any arc point.  This matches the
     * computation inside [buildUpperEyelidMesh] so callers can derive a
     * consistent [expansionFactor] without duplicating the logic.
     */
    fun arcApexRadius(lm: FloatArray, upperArcIndices: IntArray, mirrored: Boolean, aspectScale: Float = 1f): Float {
        val arcPts = indicesTo2D(lm, upperArcIndices, mirrored, aspectScale)
        val n = arcPts.size / 2
        if (n < 2) return 0.001f
        val cx = (arcPts[0] + arcPts[(n - 1) * 2]) * 0.5f
        val cy = (arcPts[1] + arcPts[(n - 1) * 2 + 1]) * 0.5f
        var apex = 0f
        for (i in 0 until n) {
            val dx = arcPts[i * 2] - cx
            val dy = arcPts[i * 2 + 1] - cy
            val d = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
            if (d > apex) apex = d
        }
        return apex.coerceAtLeast(0.001f)
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /**
     * Convert landmark index to NDC x, applying mirror and an optional
     * [aspectScale] correction.
     *
     * [aspectScale] compensates for any remaining difference between the
     * analysis-frame aspect ratio and the GL-viewport aspect ratio so that the
     * horizontal makeup positions stay aligned with the camera image.  After
     * removing the ViewPort from CameraManager both the analysis frame and the
     * camera preview cover the same sensor region, so the linear [0,1]→[-1,1]
     * mapping is already correct and [aspectScale] should be 1.0 (the default).
     * The parameter is kept as a hook for devices where the camera HAL still
     * crops the preview to a different field of view than the analysis stream.
     */
    fun lmX(lm: FloatArray, idx: Int, mirrored: Boolean, aspectScale: Float = 1f): Float {
        val nx = lm[idx * 3]           // normalised 0-1
        val mirrX = if (mirrored) 1f - nx else nx
        return (mirrX * 2f - 1f) * aspectScale  // → NDC (with optional aspect correction)
    }

    /** Convert landmark index to NDC y (flip top→bottom to bottom→top). */
    fun lmY(lm: FloatArray, idx: Int): Float {
        return 1f - lm[idx * 3 + 1] * 2f  // → NDC (flip Y)
    }

    private fun indicesTo2D(
        lm: FloatArray,
        indices: IntArray,
        mirrored: Boolean,
        aspectScale: Float = 1f,
    ): FloatArray {
        val out = FloatArray(indices.size * 2)
        for (i in indices.indices) {
            out[i * 2]     = lmX(lm, indices[i], mirrored, aspectScale)
            out[i * 2 + 1] = lmY(lm, indices[i])
        }
        return out
    }
}

private operator fun MutableList<Float>.plusAssign(arr: List<Float>) { addAll(arr) }
