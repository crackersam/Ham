package com.ham.app.face

/**
 * 1-D Kalman filter with velocity modelling – ported from the web app's
 * JavaScript implementation for identical smoothing behaviour.
 */
class KalmanFilter1D(
    private val processNoise: Float = 0.02f,
    private val measurementNoise: Float = 0.1f,
) {
    private var x = 0f
    private var p = 1f

    fun update(measurement: Float): Float {
        // Predict
        p += processNoise

        // Update
        val k = p / (p + measurementNoise)
        x += k * (measurement - x)
        p = (1f - k) * p

        return x
    }

    fun reset() {
        x = 0f; p = 1f
    }
}

/** Per-landmark (x, y, z) Kalman smoother for 478 MediaPipe face points. */
class LandmarkKalmanFilter(numLandmarks: Int) {
    private val filtersX = Array(numLandmarks) { KalmanFilter1D(0.02f, 0.1f) }
    private val filtersY = Array(numLandmarks) { KalmanFilter1D(0.02f, 0.1f) }
    private val filtersZ = Array(numLandmarks) { KalmanFilter1D(0.02f, 0.15f) }

    /** Returns a new array of filtered points. */
    fun filter(points: FloatArray): FloatArray {
        // points is laid out as [x0,y0,z0, x1,y1,z1, …]
        val count = points.size / 3
        val result = FloatArray(points.size)
        for (i in 0 until count) {
            val base = i * 3
            result[base]     = filtersX[i].update(points[base])
            result[base + 1] = filtersY[i].update(points[base + 1])
            result[base + 2] = filtersZ[i].update(points[base + 2])
        }
        return result
    }

    fun reset() {
        filtersX.forEach { it.reset() }
        filtersY.forEach { it.reset() }
        filtersZ.forEach { it.reset() }
    }
}

/**
 * Extrapolates landmark positions between MediaPipe inference completions so
 * the renderer always uses the freshest possible estimate rather than a
 * potentially 1-3 frame stale snapshot.
 *
 * On every MediaPipe result [update] computes a per-landmark velocity vector.
 * On every GL frame [predict] returns the last known positions advanced by
 * that velocity × elapsed time, capped at ~2 camera frames to avoid
 * over-extrapolation when the face is temporarily occluded.
 */
class LandmarkPredictor(private val numLandmarks: Int = 478) {
    @Volatile private var snapLandmarks: FloatArray? = null
    @Volatile private var snapTimeMs: Long = 0
    @Volatile private var velocity: FloatArray? = null // NDC units per millisecond

    @Synchronized
    fun update(landmarks: FloatArray, timestampMs: Long) {
        val prev = snapLandmarks
        val prevTs = snapTimeMs
        if (prev != null) {
            val dt = (timestampMs - prevTs).toFloat()
            if (dt in 1f..500f) {
                val vel = FloatArray(landmarks.size)
                for (i in vel.indices) vel[i] = (landmarks[i] - prev[i]) / dt
                velocity = vel
            }
        }
        snapLandmarks = landmarks.copyOf()
        snapTimeMs = timestampMs
    }

    @Synchronized
    fun predict(currentTimeMs: Long): FloatArray? {
        val lm = snapLandmarks ?: return null
        val vel = velocity ?: return lm
        val dt = (currentTimeMs - snapTimeMs).coerceIn(0L, 66L).toFloat()
        if (dt <= 0f) return lm
        val out = FloatArray(lm.size)
        for (i in out.indices) out[i] = lm[i] + vel[i] * dt
        return out
    }

    @Synchronized
    fun reset() {
        snapLandmarks = null
        snapTimeMs = 0
        velocity = null
    }
}

/**
 * Detects per-frame landmark motion to drive adaptive smoothing.
 * Returns a value in [0, 1] where 0 = no motion, 1 = fast motion.
 */
class MotionDetector {
    private var previous: FloatArray? = null

    fun detect(current: FloatArray): Float {
        val prev = previous ?: run {
            previous = current.copyOf()
            return 0f
        }
        val count = current.size / 3
        var total = 0f
        for (i in 0 until count) {
            val b = i * 3
            val dx = current[b] - prev[b]
            val dy = current[b + 1] - prev[b + 1]
            total += Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
        }
        previous = current.copyOf()
        return minOf(total / count * 5f, 1f)
    }

    fun reset() { previous = null }
}
