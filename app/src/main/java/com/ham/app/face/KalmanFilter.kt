package com.ham.app.face

/**
 * Extrapolates landmark positions between MediaPipe inference completions so
 * the renderer always uses the freshest possible estimate rather than a
 * potentially stale snapshot.
 *
 * Uses linear (constant-velocity) extrapolation:
 *   position ≈ x + v·dt
 *
 * Velocity is smoothed with an exponential moving average (α = 0.65).
 * A higher alpha (0.65 vs the previous 0.35) means velocity converges to
 * the true value in ~2 frames rather than ~6 frames, so prediction catches
 * up within 66ms when the face starts or stops moving.  MediaPipe LIVE_STREAM
 * already applies its own internal one-euro filter before the result callback
 * fires, so the raw landmarks fed here are already stable — no additional
 * position smoothing is needed or wanted (adding it introduced 100ms+ of lag).
 *
 * Extrapolation is capped at 100ms.
 */
class LandmarkPredictor(private val numLandmarks: Int = 478) {
    @Volatile private var snapLandmarks: FloatArray? = null
    @Volatile private var snapTimeMs:   Long = 0
    @Volatile private var velocity:     FloatArray? = null  // units per ms (EMA-smoothed)

    @Synchronized
    fun update(landmarks: FloatArray, timestampMs: Long) {
        val prev   = snapLandmarks
        val prevTs = snapTimeMs

        if (prev != null) {
            val dt = (timestampMs - prevTs).toFloat()
            if (dt in 1f..500f) {
                val oldVel = velocity
                val newVel = FloatArray(landmarks.size)
                for (i in newVel.indices) newVel[i] = (landmarks[i] - prev[i]) / dt

                // EMA smoothing on velocity (α = 0.65): converges to the true
                // velocity within ~2 frames rather than ~6, so the predictor
                // tracks motion starts and stops without a long ramp-up period.
                if (oldVel != null) {
                    for (i in newVel.indices)
                        newVel[i] = 0.65f * newVel[i] + 0.35f * oldVel[i]
                }
                velocity = newVel
            }
        }

        snapLandmarks = landmarks.copyOf()
        snapTimeMs    = timestampMs
    }

    @Synchronized
    fun predict(currentTimeMs: Long): FloatArray? {
        val lm  = snapLandmarks ?: return null
        val vel = velocity       ?: return lm

        val dt = (currentTimeMs - snapTimeMs).coerceIn(0L, 150L).toFloat()
        if (dt <= 0f) return lm

        val out = FloatArray(lm.size)
        for (i in out.indices) out[i] = lm[i] + vel[i] * dt
        return out
    }

    @Synchronized
    fun reset() {
        snapLandmarks = null
        snapTimeMs    = 0
        velocity      = null
    }
}
