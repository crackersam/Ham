package com.ham.app.render

import androidx.compose.ui.graphics.Color

enum class ContourBlendMode { MULTIPLY, SOFT_LIGHT }

/**
 * Contour pipeline parameters.
 *
 * This matches the requested rendering stack:
 * mask(per-region) → gradient → blur → multiply/softlight composite.
 */
data class ContourParams(
    // Master enable/intensity (0..1). This multiplies all region strengths.
    val intensity: Float = 1.0f,

    // Skin-derived shadow shade (computed per frame from sampled skin tone).
    val shade: Color,
    // 0..1 cool-toned correction: reduce redness slightly for a pro “shadow” read.
    val coolTone: Float = 0.65f,

    // Per-region strengths (0..1), tuned to be instantly visible but still blended.
    // Defaults are the requested starting values.
    val cheekContour: Float = 0.60f,
    val jawContour: Float = 0.50f,
    val noseContour: Float = 0.40f,
    val chinContour: Float = 0.50f,

    // Blur strength multiplier applied to the face-scaled blur radius.
    val softness: Float = 1.0f,
    // Face clip-mask erosion factor in pixels, expressed as a fraction of face width.
    val faceErosionScale: Float = 0.01f,

    // Composite mode.
    val blendMode: ContourBlendMode = ContourBlendMode.MULTIPLY,

    // Landmark temporal smoothing: smoothed = lerp(prev, curr, 0.20)
    // NOTE: We apply this to derived anchor points (not all 478 landmarks).
    val landmarkSmoothingT: Float = 0.20f,

    // Mask texture temporal smoothing (EMA weight of current frame in 0..1).
    val temporalSmoothing: Float = 0.60f,
)

