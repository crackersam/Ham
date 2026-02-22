package com.ham.app.data

import androidx.compose.ui.graphics.Color

data class MakeupStyle(
    val id: String,
    val name: String,
    val lipColor: Color,
    val eyeshadowColor: Color,
    val blushColor: Color,
    val linerColor: Color,
    val lipAlpha: Float,
    val eyeshadowAlpha: Float,
    val blushAlpha: Float,
    val linerAlpha: Float,
    val highlightColor: Color = Color(0xFFFFF0E0),
    val highlightAlpha: Float = 0f,
    val foundationAlpha: Float = 0f,
)

val MAKEUP_STYLES = listOf(
    MakeupStyle(
        id = "none",
        name = "None",
        lipColor = Color.Transparent,
        eyeshadowColor = Color.Transparent,
        blushColor = Color.Transparent,
        linerColor = Color.Transparent,
        lipAlpha = 0f,
        eyeshadowAlpha = 0f,
        blushAlpha = 0f,
        linerAlpha = 0f,
    ),
    MakeupStyle(
        id = "soft-day",
        name = "Soft Day",
        lipColor = Color(0xFFc97888),
        eyeshadowColor = Color(0xFFd4b5d8),
        blushColor = Color(0xFFe8a5a8),
        linerColor = Color(0xFF4d3d3f),
        lipAlpha = 0.68f,
        eyeshadowAlpha = 0.50f,
        blushAlpha = 0.48f,
        linerAlpha = 0.82f,
        foundationAlpha = 0.28f,
    ),
    MakeupStyle(
        id = "classic-evening",
        name = "Classic Evening",
        lipColor = Color(0xFFa63852),
        eyeshadowColor = Color(0xFF8e6b9f),
        blushColor = Color(0xFFd88090),
        linerColor = Color(0xFF2b1f26),
        lipAlpha = 0.85f,
        eyeshadowAlpha = 0.65f,
        blushAlpha = 0.60f,
        linerAlpha = 0.98f,
        foundationAlpha = 0.32f,
    ),
    MakeupStyle(
        id = "bridal-glow",
        name = "Bridal Glow",
        lipColor = Color(0xFFd98e8d),
        eyeshadowColor = Color(0xFFdbb5a0),
        blushColor = Color(0xFFf0b5b0),
        linerColor = Color(0xFF4a3a3e),
        lipAlpha = 0.75f,
        eyeshadowAlpha = 0.52f,
        blushAlpha = 0.58f,
        linerAlpha = 0.85f,
        highlightColor = Color(0xFFFFF8F0),
        highlightAlpha = 0.3f,
        foundationAlpha = 0.38f,
    ),
    MakeupStyle(
        id = "editorial",
        name = "Editorial",
        lipColor = Color(0xFFb8325f),
        eyeshadowColor = Color(0xFF9e5496),
        blushColor = Color(0xFFde6f8a),
        linerColor = Color(0xFF1a1418),
        lipAlpha = 0.95f,
        eyeshadowAlpha = 0.72f,
        blushAlpha = 0.68f,
        linerAlpha = 1.0f,
        foundationAlpha = 0.30f,
    ),
)

// ─── MediaPipe FaceLandmarker 478-point landmark indices ───────────────────

/**
 * Face silhouette oval — 36 points, clockwise from forehead centre.
 * Used as the foundation mesh boundary so skin-smoothing and colour
 * correction are confined precisely to the face.
 */
val FACE_OVAL = intArrayOf(
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109
)

/** Outer lip ring – clockwise */
val LIPS_OUTER = intArrayOf(
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
)

/** Inner lip ring (mouth opening) – clockwise */
val LIPS_INNER = intArrayOf(
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
)

/** Cupid's bow highlight points */
val LIP_CUPID_BOW = intArrayOf(0, 37, 267)

/** Left eye socket / eyeshadow region */
val LEFT_EYE_SOCKET = intArrayOf(
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246
)

/** Right eye socket / eyeshadow region */
val RIGHT_EYE_SOCKET = intArrayOf(
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466
)

/**
 * Left upper eyelid arc — inner corner → upper lash line → outer corner.
 * Contains NO lower-lid points so the eyeshadow mesh built from this arc
 * sits entirely above the lash line and never covers the eyeball.
 */
val LEFT_UPPER_LID_ARC = intArrayOf(133, 173, 157, 158, 159, 160, 161, 246, 33)

/**
 * Right upper eyelid arc — inner corner → upper lash line → outer corner.
 * Mirror of LEFT_UPPER_LID_ARC for the right eye.
 */
val RIGHT_UPPER_LID_ARC = intArrayOf(362, 398, 384, 385, 386, 387, 388, 466, 263)

/** Left eyeliner path (lower lash line) */
val LEFT_LINER = intArrayOf(33, 160, 159, 158, 157, 173, 133)

/** Right eyeliner path (lower lash line) */
val RIGHT_LINER = intArrayOf(263, 387, 386, 385, 384, 398, 362)

/** Left upper-lash liner */
val LEFT_UPPER_LINER = intArrayOf(246, 161, 160, 159, 158, 157, 173)

/** Right upper-lash liner */
val RIGHT_UPPER_LINER = intArrayOf(466, 388, 387, 386, 385, 384, 398)

/** Left cheek blush region */
val LEFT_CHEEK = intArrayOf(50, 123, 116, 117, 118, 101, 205, 36, 187, 147)

/** Right cheek blush region */
val RIGHT_CHEEK = intArrayOf(280, 352, 346, 347, 348, 330, 425, 266, 411, 376)

/** Left eyebrow */
val LEFT_BROW = intArrayOf(46, 53, 52, 65, 55, 70, 63, 105, 66, 107)

/** Right eyebrow */
val RIGHT_BROW = intArrayOf(276, 283, 282, 295, 285, 300, 293, 334, 296, 336)

/** Nose bridge for highlight */
val NOSE_BRIDGE = intArrayOf(168, 6, 197, 195, 5)

/** Forehead highlight / contour anchor points */
val FOREHEAD_CENTER = intArrayOf(10, 151, 9, 8)

/** Key point indices for single-point lookups */
object LandmarkIndex {
    const val NOSE_TIP = 1
    const val CHIN = 152
    const val LEFT_EYE_OUTER = 33
    const val LEFT_EYE_INNER = 133
    const val LEFT_EYE_LOWER = 145
    const val RIGHT_EYE_OUTER = 263
    const val RIGHT_EYE_INNER = 362
    const val RIGHT_EYE_LOWER = 374
    const val LIP_TOP_CENTER = 0
    const val LIP_BOTTOM_CENTER = 17
    const val LIP_LEFT = 61
    const val LIP_RIGHT = 291
    const val LEFT_CHEEK_CENTER = 205
    const val RIGHT_CHEEK_CENTER = 425
}
