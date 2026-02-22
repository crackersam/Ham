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
     * Builds a feathered fan mesh from an unordered/contour-ish set of points by
     * sorting them around the centroid. Useful for soft filled regions like brows.
     */
    fun buildSortedFanMesh(
        lm: FloatArray,
        indices: IntArray,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
    ): FloatArray {
        val pts = indicesTo2D(lm, indices, isMirrored, aspectScale)
        return buildSortedFanMesh2D(pts)
    }

    /**
     * Like [buildSortedFanMesh] but takes explicit NDC points: [x0,y0,x1,y1,...].
     */
    fun buildSortedFanMesh2D(ndcPts: FloatArray): FloatArray {
        val n = ndcPts.size / 2
        if (n < 3) return FloatArray(0)

        var cx = 0f; var cy = 0f
        for (i in 0 until n) { cx += ndcPts[i * 2]; cy += ndcPts[i * 2 + 1] }
        cx /= n.toFloat(); cy /= n.toFloat()

        val order = (0 until n).sortedBy { i ->
            val x = ndcPts[i * 2] - cx
            val y = ndcPts[i * 2 + 1] - cy
            Math.atan2(y.toDouble(), x.toDouble()).toFloat()
        }

        val verts = FloatArray(n * 3 * STRIDE)
        var vi = 0
        for (k in 0 until n) {
            val i0 = order[k]
            val i1 = order[(k + 1) % n]

            // Center hub
            verts[vi++] = cx; verts[vi++] = cy
            verts[vi++] = 1f; verts[vi++] = 0.5f; verts[vi++] = 0.5f

            // Edge i0
            verts[vi++] = ndcPts[i0 * 2]; verts[vi++] = ndcPts[i0 * 2 + 1]
            verts[vi++] = 0f; verts[vi++] = k.toFloat() / n.toFloat(); verts[vi++] = 0f

            // Edge i1
            verts[vi++] = ndcPts[i1 * 2]; verts[vi++] = ndcPts[i1 * 2 + 1]
            verts[vi++] = 0f; verts[vi++] = (k + 1).toFloat() / n.toFloat(); verts[vi++] = 0f
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
        return buildStrokeMesh2D(pts, widthNdc)
    }

    /**
     * Like [buildStrokeMesh], but takes an explicit polyline in NDC space:
     * [x0, y0, x1, y1, ...].
     *
     * Produces a featherable stroke by building two strips per segment:
     *   leftEdge (edgeFactor=0) ↔ centerLine (edgeFactor=1)
     *   centerLine (edgeFactor=1) ↔ rightEdge (edgeFactor=0)
     */
    fun buildStrokeMesh2D(
        ndcPts: FloatArray,
        widthNdc: Float,
    ): FloatArray {
        if (ndcPts.size < 4) return FloatArray(0) // need ≥ 2 points

        val n = ndcPts.size / 2
        val result = mutableListOf<Float>()

        fun addTri(ax: Float, ay: Float, aEf: Float,
                   bx: Float, by: Float, bEf: Float,
                   cx: Float, cy: Float, cEf: Float) {
            // regionUV is currently unused for strokes, but keep stable values
            result += floatArrayOf(ax, ay, aEf, 0f, 0f).toList()
            result += floatArrayOf(bx, by, bEf, 0.5f, 0.5f).toList()
            result += floatArrayOf(cx, cy, cEf, 1f, 1f).toList()
        }

        for (i in 0 until n - 1) {
            val x0 = ndcPts[i * 2]; val y0 = ndcPts[i * 2 + 1]
            val x1 = ndcPts[(i + 1) * 2]; val y1 = ndcPts[(i + 1) * 2 + 1]

            val dx = x1 - x0; val dy = y1 - y0
            val len = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat().coerceAtLeast(1e-6f)
            val nx = -dy / len * widthNdc
            val ny =  dx / len * widthNdc

            val l0x = x0 + nx; val l0y = y0 + ny
            val r0x = x0 - nx; val r0y = y0 - ny
            val l1x = x1 + nx; val l1y = y1 + ny
            val r1x = x1 - nx; val r1y = y1 - ny

            // Strip A: left edge -> centerline
            // Tri 1: l0, c0, l1
            addTri(l0x, l0y, 0f, x0, y0, 1f, l1x, l1y, 0f)
            // Tri 2: c0, c1, l1
            addTri(x0, y0, 1f, x1, y1, 1f, l1x, l1y, 0f)

            // Strip B: centerline -> right edge
            // Tri 3: c0, r0, c1
            addTri(x0, y0, 1f, r0x, r0y, 0f, x1, y1, 1f)
            // Tri 4: c1, r0, r1
            addTri(x1, y1, 1f, r0x, r0y, 0f, r1x, r1y, 0f)
        }

        return result.toFloatArray()
    }

    // ── Brow fill mesh (ribbon) ──────────────────────────────────────────────

    /**
     * Builds a brow fill mesh as a "ribbon" between MediaPipe's two eyebrow
     * contour polylines (5 points each, concatenated to 10 total).
     *
     * This avoids the centroid-fan triangulation used by [buildSortedFanMesh]
     * which can self-intersect for eyebrow shapes and cause pigment spill
     * below the underside edge.
     *
     * Feathering for a natural finish:
     * - Underside edge is kept dense (edgeFactor ~= 1).
     * - Upper edge fades out (edgeFactor = 0).
     * - A mid ring keeps most of the brow volume opaque, with fade confined
     *   to the upper portion.
     */
    fun buildBrowRibbonMesh(
        lm: FloatArray,
        browIndices: IntArray,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
        midT: Float = 0.35f,
    ): FloatArray {
        val half = browIndices.size / 2
        if (half < 2 || browIndices.size < 4) return FloatArray(0)

        val chainA = browIndices.copyOfRange(0, half)
        val chainB = browIndices.copyOfRange(half, browIndices.size)

        val a = indicesTo2D(lm, chainA, isMirrored, aspectScale)
        val b = indicesTo2D(lm, chainB, isMirrored, aspectScale)
        val n = minOf(a.size, b.size) / 2
        if (n < 2) return FloatArray(0)

        fun minY(pts: FloatArray): Float {
            var mn = Float.POSITIVE_INFINITY
            val m = pts.size / 2
            for (i in 0 until m) {
                val y = pts[i * 2 + 1]
                if (y < mn) mn = y
            }
            return mn
        }

        // NDC y increases upward; the underside is the *lower* polyline.
        val underside = if (minY(a) < minY(b)) a else b
        val topSide   = if (underside === a) b else a

        fun addV(
            out: MutableList<Float>,
            x: Float,
            y: Float,
            ef: Float,
            u: Float,
            v: Float,
        ) {
            out += floatArrayOf(x, y, ef, u, v).toList()
        }

        fun endFade(i: Int): Float {
            // Keep ends slightly softer but never "off".
            val t = if (n == 1) 0.5f else i.toFloat() / (n - 1).toFloat()
            val bell = Math.sin((t * Math.PI).toDouble()).toFloat().coerceIn(0f, 1f) // 0→1→0
            return (0.78f + 0.22f * bell).coerceIn(0f, 1f)
        }

        val tMid = midT.coerceIn(0.15f, 0.65f)
        val out = mutableListOf<Float>()

        for (i in 0 until n - 1) {
            val next = i + 1

            val u0 = i.toFloat() / (n - 1).toFloat()
            val u1 = next.toFloat() / (n - 1).toFloat()

            val b0x = underside[i * 2];       val b0y = underside[i * 2 + 1]
            val b1x = underside[next * 2];    val b1y = underside[next * 2 + 1]
            val t0x = topSide[i * 2];         val t0y = topSide[i * 2 + 1]
            val t1x = topSide[next * 2];      val t1y = topSide[next * 2 + 1]

            // Mid ring: keeps brow volume opaque; fade is concentrated above it.
            val m0x = b0x + (t0x - b0x) * tMid
            val m0y = b0y + (t0y - b0y) * tMid
            val m1x = b1x + (t1x - b1x) * tMid
            val m1y = b1y + (t1y - b1y) * tMid

            val ef0 = endFade(i)
            val ef1 = endFade(next)

            // Strip 1: underside (v=0) -> mid (v=0.5), both dense
            // Tri 1: b0, b1, m0
            addV(out, b0x, b0y, ef0, u0, 0f)
            addV(out, b1x, b1y, ef1, u1, 0f)
            addV(out, m0x, m0y, ef0, u0, 0.5f)
            // Tri 2: m0, b1, m1
            addV(out, m0x, m0y, ef0, u0, 0.5f)
            addV(out, b1x, b1y, ef1, u1, 0f)
            addV(out, m1x, m1y, ef1, u1, 0.5f)

            // Strip 2: mid (dense) -> top (fade out)
            // Tri 3: m0, m1, t0
            addV(out, m0x, m0y, ef0, u0, 0.5f)
            addV(out, m1x, m1y, ef1, u1, 0.5f)
            addV(out, t0x, t0y, 0f,  u0, 1f)
            // Tri 4: t0, m1, t1
            addV(out, t0x, t0y, 0f,  u0, 1f)
            addV(out, m1x, m1y, ef1, u1, 0.5f)
            addV(out, t1x, t1y, 0f,  u1, 1f)
        }

        return out.toFloatArray()
    }

    /**
     * Returns the underside brow polyline (inner→outer) in NDC space.
     * Intended for the definition stroke so the crisp line follows the
     * underside boundary rather than a centroid-derived spine.
     */
    fun buildBrowUndersidePath2D(
        lm: FloatArray,
        browIndices: IntArray,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
        targetPoints: Int = 7,
    ): FloatArray {
        val half = browIndices.size / 2
        if (half < 2 || browIndices.size < 4) return FloatArray(0)
        val chainA = browIndices.copyOfRange(0, half)
        val chainB = browIndices.copyOfRange(half, browIndices.size)
        val a = indicesTo2D(lm, chainA, isMirrored, aspectScale)
        val b = indicesTo2D(lm, chainB, isMirrored, aspectScale)
        val n = minOf(a.size, b.size) / 2
        if (n < 2) return FloatArray(0)

        fun minY(pts: FloatArray): Float {
            var mn = Float.POSITIVE_INFINITY
            val m = pts.size / 2
            for (i in 0 until m) {
                val y = pts[i * 2 + 1]
                if (y < mn) mn = y
            }
            return mn
        }
        val underside = if (minY(a) < minY(b)) a else b

        // Smooth with a simple 3-point moving average.
        val sm = FloatArray(n * 2)
        for (i in 0 until n) {
            val i0 = (i - 1).coerceAtLeast(0)
            val i1 = i
            val i2 = (i + 1).coerceAtMost(n - 1)
            sm[i * 2]     = (underside[i0 * 2]     + underside[i1 * 2]     + underside[i2 * 2])     / 3f
            sm[i * 2 + 1] = (underside[i0 * 2 + 1] + underside[i1 * 2 + 1] + underside[i2 * 2 + 1]) / 3f
        }

        val outCount = targetPoints.coerceIn(2, n)
        val out = FloatArray(outCount * 2)
        for (k in 0 until outCount) {
            val idx = ((k.toFloat() / (outCount - 1).toFloat()) * (n - 1).toFloat()).toInt()
            out[k * 2]     = sm[idx * 2]
            out[k * 2 + 1] = sm[idx * 2 + 1]
        }
        return out
    }

    /**
     * Derive a single brow spine polyline (inner→outer) from the brow landmark
     * set. Intended for ONE subtle definition stroke (not a contour trace).
     *
     * Returns NDC points: [x0,y0,x1,y1,...].
     */
    fun buildBrowSpine2D(
        lm: FloatArray,
        browIndices: IntArray,
        isMirrored: Boolean = true,
        aspectScale: Float = 1f,
        noseIdx: Int = LandmarkIndex.NOSE_TIP,
        targetPoints: Int = 7,
    ): FloatArray {
        val pts = indicesTo2D(lm, browIndices, isMirrored, aspectScale)
        val n = pts.size / 2
        if (n < 2) return FloatArray(0)

        val noseX = lmX(lm, noseIdx, isMirrored, aspectScale)

        // Find inner/outer endpoints based on distance from the nose.
        var innerI = 0
        var outerI = 0
        var minD = Float.MAX_VALUE
        var maxD = -Float.MAX_VALUE
        for (i in 0 until n) {
            val d = kotlin.math.abs(pts[i * 2] - noseX)
            if (d < minD) { minD = d; innerI = i }
            if (d > maxD) { maxD = d; outerI = i }
        }

        val ix = pts[innerI * 2]; val iy = pts[innerI * 2 + 1]
        val ox = pts[outerI * 2]; val oy = pts[outerI * 2 + 1]

        var dirX = ox - ix
        var dirY = oy - iy
        val dirLen = Math.sqrt((dirX * dirX + dirY * dirY).toDouble()).toFloat().coerceAtLeast(1e-6f)
        dirX /= dirLen; dirY /= dirLen

        data class P(val x: Float, val y: Float, val t: Float)
        val ordered = ArrayList<P>(n)
        for (i in 0 until n) {
            val px = pts[i * 2]; val py = pts[i * 2 + 1]
            val t = (px - ix) * dirX + (py - iy) * dirY
            ordered.add(P(px, py, t))
        }
        ordered.sortBy { it.t }

        // Smooth with a simple 3-point moving average.
        val sm = ArrayList<P>(ordered.size)
        for (i in ordered.indices) {
            val p0 = ordered[(i - 1).coerceAtLeast(0)]
            val p1 = ordered[i]
            val p2 = ordered[(i + 1).coerceAtMost(ordered.lastIndex)]
            val sx = (p0.x + p1.x + p2.x) / 3f
            val sy = (p0.y + p1.y + p2.y) / 3f
            sm.add(P(sx, sy, p1.t))
        }

        // Downsample uniformly to targetPoints, preserving endpoints.
        val outCount = targetPoints.coerceIn(2, sm.size)
        val out = FloatArray(outCount * 2)
        for (k in 0 until outCount) {
            val idx = ((k.toFloat() / (outCount - 1).toFloat()) * (sm.size - 1).toFloat()).toInt()
            out[k * 2] = sm[idx].x
            out[k * 2 + 1] = sm[idx].y
        }
        return out
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
            val c0 = Math.cos(a0.toDouble()).toFloat()
            val s0 = Math.sin(a0.toDouble()).toFloat()
            verts[vi++] = centerX + c0 * radiusX
            verts[vi++] = centerY + s0 * radiusY
            // Radial UV: maps rim to the unit circle around (0.5, 0.5).
            verts[vi++] = 0f; verts[vi++] = 0.5f + 0.5f * c0; verts[vi++] = 0.5f + 0.5f * s0
            // Edge vertex i+1
            val c1 = Math.cos(a1.toDouble()).toFloat()
            val s1 = Math.sin(a1.toDouble()).toFloat()
            verts[vi++] = centerX + c1 * radiusX
            verts[vi++] = centerY + s1 * radiusY
            verts[vi++] = 0f; verts[vi++] = 0.5f + 0.5f * c1; verts[vi++] = 0.5f + 0.5f * s1
        }
        return verts
    }

    // ── Lash mesh ────────────────────────────────────────────────────────────

    /**
     * Builds a natural-looking upper-lash mesh: small tapered spikes along the
     * upper lash line, pointing outward from the eye center.
     *
     * Uses triangle spikes with edgeFactor=1 at the base and 0 at the tip so
     * the shader tapers them naturally.
     */
    fun buildUpperLashesMesh(
        lm: FloatArray,
        upperLinerIndices: IntArray,
        isMirrored: Boolean = true,
        lengthNdc: Float,
        thicknessNdc: Float,
        aspectScale: Float = 1f,
        subdivisionsPerSegment: Int = 3,
    ): FloatArray {
        val arc = indicesTo2D(lm, upperLinerIndices, isMirrored, aspectScale)
        val n = arc.size / 2
        if (n < 2) return FloatArray(0)

        // Eye center as mean of arc points.
        var cx = 0f; var cy = 0f
        for (i in 0 until n) { cx += arc[i * 2]; cy += arc[i * 2 + 1] }
        cx /= n.toFloat(); cy /= n.toFloat()

        val segCount = n - 1
        val subs = subdivisionsPerSegment.coerceIn(1, 6)

        val result = mutableListOf<Float>()

        fun addV(x: Float, y: Float, ef: Float) {
            // Keep regionUV stable; not used for lashes.
            result += floatArrayOf(x, y, ef, 0.5f, 0.5f).toList()
        }

        for (si in 0 until segCount) {
            val x0 = arc[si * 2];     val y0 = arc[si * 2 + 1]
            val x1 = arc[(si + 1) * 2]; val y1 = arc[(si + 1) * 2 + 1]

            var tx = x1 - x0
            var ty = y1 - y0
            val tLen = Math.sqrt((tx * tx + ty * ty).toDouble()).toFloat().coerceAtLeast(1e-6f)
            tx /= tLen; ty /= tLen

            for (s in 0 until subs) {
                val t = (s.toFloat() + 0.5f) / subs.toFloat()
                val px = x0 + (x1 - x0) * t
                val py = y0 + (y1 - y0) * t

                // Normalized position along lid for bell-curve lash length.
                val u = (si.toFloat() + t) / segCount.toFloat()
                val bell = Math.sin((u * Math.PI).toDouble()).toFloat().coerceIn(0f, 1f)
                val lashLen = lengthNdc * (0.35f + 0.65f * bell)

                // Point lashes outward from eye center.
                var dx = px - cx
                var dy = py - cy
                val dLen = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat().coerceAtLeast(1e-6f)
                dx /= dLen; dy /= dLen

                val tipX = px + dx * lashLen
                val tipY = py + dy * lashLen

                val baseHalf = thicknessNdc * (0.75f + 0.45f * bell)
                val b0x = px - tx * baseHalf
                val b0y = py - ty * baseHalf
                val b1x = px + tx * baseHalf
                val b1y = py + ty * baseHalf

                // Triangle: base0, base1, tip
                // Strong at base (ef=1), feather to tip (ef=0).
                addV(b0x, b0y, 1f)
                addV(b1x, b1y, 1f)
                addV(tipX, tipY, 0f)
            }
        }

        return result.toFloatArray()
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
